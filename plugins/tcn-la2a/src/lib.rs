//! LA2A-style neural compressor wrapper.
//!
//! This crate is a thin shim over [`tcn_plugin_core`]: the heavy lifting
//! (model load, FiLM conditioning, sample-rate DSP) lives there. Here we only
//! declare the two knobs the LA2A model was trained with (`peak_reduction`,
//! `limit`) and map them to the flat `&[f32]` the model expects.
//!
//! Adding a new model = copy this crate, update `NPARAMS`, `Params`, and the
//! `update_conditioning` argument list.

use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, EguiState};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use tcn_plugin_core::{load_model, TcnModel};

pub mod gui;
use gui::MeterState;

/// Baked-in default model, included at compile time so a freshly built plugin
/// works out of the box. Runtime overrides (env var or bundle Resources/)
/// still win if set. Kept at `plugins/tcn-la2a/assets/tcn.json`; replace and
/// rebuild to ship a new default.
const DEFAULT_MODEL_JSON: &str = include_str!("../assets/tcn.json");

/// Env var used to override the model path at runtime. Per-plugin so two
/// wrappers installed side by side don't collide.
const MODEL_ENV_OVERRIDE: &str = "TCN_LA2A_MODEL";

/// Number of conditioning parameters the LA2A model was trained with. Matched
/// against the loaded JSON at init — a 3-knob model dropped in here will fail
/// loudly instead of silently producing garbage conditioning.
const NPARAMS: usize = 2;

pub struct TcnLa2a {
    params: Arc<TcnLa2aParams>,
    model: Option<TcnModel>,
    sample_rate: f32,
    /// Scratch passed to `update_conditioning`. Reused each buffer to avoid
    /// allocating on the audio thread.
    cond_scratch: [f32; NPARAMS],
    /// Meter peaks written by the audio thread, read by the GUI.
    meters: Arc<MeterState>,
}

#[derive(Params)]
pub struct TcnLa2aParams {
    #[persist = "editor-state"]
    pub editor_state: Arc<EguiState>,

    #[id = "peak_red"]
    pub peak_reduction: FloatParam,

    #[id = "limit"]
    pub limit: BoolParam,

    #[id = "makeup_db"]
    pub makeup_gain_db: FloatParam,
}

impl Default for TcnLa2aParams {
    fn default() -> Self {
        Self {
            editor_state: EguiState::from_size(gui::GUI_WIDTH, gui::GUI_HEIGHT),
            peak_reduction: FloatParam::new(
                "Peak Reduction",
                0.5,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(50.0)),
            limit: BoolParam::new("Limit", false),
            makeup_gain_db: FloatParam::new(
                "Makeup Gain",
                0.0,
                FloatRange::Linear { min: 0.0, max: 24.0 },
            )
            .with_unit(" dB"),
        }
    }
}

impl Default for TcnLa2a {
    fn default() -> Self {
        Self {
            params: Arc::new(TcnLa2aParams::default()),
            model: None,
            sample_rate: 44100.0,
            cond_scratch: [0.0; NPARAMS],
            meters: Arc::new(MeterState::default()),
        }
    }
}

impl Plugin for TcnLa2a {
    const NAME: &'static str = "micro-TCN LA2A";
    const VENDOR: &'static str = "micro-tcn";
    const URL: &'static str = "https://github.com/stevemurr/micro-tcn";
    const EMAIL: &'static str = "noreply@example.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(1),
        main_output_channels: NonZeroU32::new(1),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        let params = self.params.clone();
        let meters = self.meters.clone();
        create_egui_editor(
            self.params.editor_state.clone(),
            (),
            |_, _| {},
            move |ctx, setter, _state| {
                gui::draw_ui(ctx, setter, &params, &meters);
            },
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;

        // Pre-warm smoothers so the first buffer doesn't see garbage values.
        self.params
            .peak_reduction
            .smoothed
            .reset(self.params.peak_reduction.value());

        // Skip the 1.1 MB JSON parse if a model is already loaded (DAW may
        // call initialize() repeatedly on sample-rate changes or bounce).
        if self.model.is_some() {
            return true;
        }

        let (model, source) = match load_model(MODEL_ENV_OVERRIDE, DEFAULT_MODEL_JSON) {
            Ok(ok) => ok,
            Err(e) => {
                nih_log!("failed to load TCN model: {}", e);
                return false;
            }
        };

        if let Err(e) = model.require_nparams(NPARAMS) {
            nih_log!(
                "model mismatch: {} — this wrapper only accepts LA2A-shaped models",
                e
            );
            return false;
        }

        nih_log!("loaded TCN model from {}", source);
        self.model = Some(model);
        true
    }

    fn reset(&mut self) {
        if let Some(m) = self.model.as_mut() {
            m.reset();
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let model = match self.model.as_mut() {
            Some(m) => m,
            None => return ProcessStatus::Normal,
        };

        // Update conditioning once per buffer. Cheap — touches the adaptor MLP
        // + per-block scale/shift vectors, never the per-sample hot path. Using
        // .value() (not .smoothed.next()) because next() advances by one sample
        // per call and we call it once per buffer — would take forever to
        // actually reach the target under automation.
        self.cond_scratch[0] = if self.params.limit.value() { 1.0 } else { 0.0 };
        self.cond_scratch[1] = self.params.peak_reduction.value();
        model.update_conditioning(&self.cond_scratch);

        let makeup_db = self.params.makeup_gain_db.value();
        let makeup_gain = if makeup_db > 0.01 { 10.0f32.powf(makeup_db / 20.0) } else { 1.0 };

        let mut in_peak = 0.0f32;
        let mut raw_out_peak = 0.0f32;

        for channel_samples in buffer.iter_samples() {
            for sample in channel_samples {
                in_peak = in_peak.max(sample.abs());
                *sample = model.process_sample(*sample);
                raw_out_peak = raw_out_peak.max(sample.abs());
                *sample *= makeup_gain;
            }
        }

        // Fast-attack / slow-decay meter updates.
        const DECAY: f32 = 0.95;
        let update_peak = |atom: &AtomicU32, new: f32| {
            let old = f32::from_bits(atom.load(Ordering::Relaxed));
            atom.store((if new >= old { new } else { old * DECAY }).to_bits(), Ordering::Relaxed);
        };
        update_peak(&self.meters.input_peak, in_peak);
        update_peak(&self.meters.output_peak, raw_out_peak * makeup_gain);
        let gr_db = if in_peak > 1e-5 && raw_out_peak > 1e-5 {
            20.0 * (raw_out_peak / in_peak).log10()
        } else {
            0.0
        };
        self.meters.gain_reduction.store(gr_db.to_bits(), Ordering::Relaxed);

        ProcessStatus::Normal
    }
}

impl ClapPlugin for TcnLa2a {
    const CLAP_ID: &'static str = "com.microtcn.tcn-la2a";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Neural dynamic range compressor modeled on LA2A.");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Mono,
        ClapFeature::Compressor,
    ];
}

impl Vst3Plugin for TcnLa2a {
    const VST3_CLASS_ID: [u8; 16] = *b"microtcnTCNLa2aV";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(TcnLa2a);
nih_export_vst3!(TcnLa2a);

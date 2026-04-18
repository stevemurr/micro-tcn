//! micro-TCN CLAP + VST3 plugin.
//!
//! Loads a `tcn.json` exported by `microtcn export` and runs the TCN in the
//! audio callback. Parameters map to the model's conditioning: `peak_reduction`
//! is the learned peak-reduction axis (0..1), `limit` selects compress (0) or
//! limit (1) mode.
//!
//! The DSP lives in `model.rs`. This file is pure nih-plug plumbing.

use nih_plug::prelude::*;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

mod model;
use model::TcnModel;

pub struct TcnClap {
    params: Arc<TcnClapParams>,
    model: Option<TcnModel>,
    sample_rate: f32,
}

#[derive(Params)]
pub struct TcnClapParams {
    #[id = "peak_red"]
    pub peak_reduction: FloatParam,

    #[id = "limit"]
    pub limit: BoolParam,
}

impl Default for TcnClapParams {
    fn default() -> Self {
        Self {
            peak_reduction: FloatParam::new(
                "Peak Reduction",
                0.5,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(50.0)),
            limit: BoolParam::new("Limit", false),
        }
    }
}

impl Default for TcnClap {
    fn default() -> Self {
        Self {
            params: Arc::new(TcnClapParams::default()),
            model: None,
            sample_rate: 44100.0,
        }
    }
}

fn locate_model_json() -> Option<PathBuf> {
    if let Ok(p) = env::var("TCN_CLAP_MODEL") {
        return Some(PathBuf::from(p));
    }
    // Fallback: look next to the plugin binary for tcn.json.
    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("tcn.json");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

impl Plugin for TcnClap {
    const NAME: &'static str = "micro-TCN";
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

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;

        match locate_model_json() {
            Some(path) => match TcnModel::load_from_json(&path) {
                Ok(m) => {
                    nih_log!("loaded TCN model from {}", path.display());
                    self.model = Some(m);
                    true
                }
                Err(e) => {
                    nih_log!("failed to load {}: {}", path.display(), e);
                    false
                }
            },
            None => {
                nih_log!(
                    "no TCN model found. Set TCN_CLAP_MODEL or place tcn.json next to the plugin."
                );
                false
            }
        }
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

        // Update conditioning on param change. Cheap — only touches the adaptor
        // MLP + per-block scale/shift vectors, never in the per-sample hot path.
        let peak_red = self.params.peak_reduction.smoothed.next();
        let limit = if self.params.limit.value() { 1.0_f32 } else { 0.0_f32 };
        model.update_conditioning(limit, peak_red);

        for channel_samples in buffer.iter_samples() {
            for sample in channel_samples {
                *sample = model.process_sample(*sample);
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for TcnClap {
    const CLAP_ID: &'static str = "com.microtcn.tcn-clap";
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

impl Vst3Plugin for TcnClap {
    const VST3_CLASS_ID: [u8; 16] = *b"microtcnTCNClapV";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nih_export_clap!(TcnClap);
nih_export_vst3!(TcnClap);

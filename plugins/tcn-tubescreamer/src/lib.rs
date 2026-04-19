//! Tube-screamer distortion wrapper.
//!
//! The model was captured at a single fixed knob position (the EGFxSet
//! TubeScreamer subset records one setting per pickup), so `nparams = 0` and
//! the plugin exposes no model-conditioning knobs. FiLM becomes a learned
//! per-block bias at training time, which the runtime replays each buffer.
//!
//! If you re-record the pedal at multiple Drive/Tone/Level positions and
//! retrain, this crate is the right place to add the knobs: bump `NPARAMS`,
//! declare `FloatParam`s in `TcnTubeScreamerParams`, and fill `cond_scratch`
//! in the same order the dataset was captured.

use nih_plug::prelude::*;
use std::sync::Arc;

use tcn_plugin_core::{load_model, TcnModel};

const DEFAULT_MODEL_JSON: &str = include_str!("../assets/tcn.json");
const MODEL_ENV_OVERRIDE: &str = "TCN_TUBESCREAMER_MODEL";

/// Conditioning parameters the model was trained with. Zero for the current
/// fixed-setting capture; increase and add matching `FloatParam`s if retraining
/// on a knob-swept dataset.
const NPARAMS: usize = 0;

pub struct TcnTubeScreamer {
    params: Arc<TcnTubeScreamerParams>,
    model: Option<TcnModel>,
    sample_rate: f32,
    cond_scratch: [f32; NPARAMS],
}

/// No knobs — the model was trained at a single fixed drive/tone/level.
/// Kept as a named struct rather than `()` so future knob additions have an
/// obvious place to land.
#[derive(Params, Default)]
pub struct TcnTubeScreamerParams {}

impl Default for TcnTubeScreamer {
    fn default() -> Self {
        Self {
            params: Arc::new(TcnTubeScreamerParams::default()),
            model: None,
            sample_rate: 44100.0,
            cond_scratch: [0.0; NPARAMS],
        }
    }
}

impl Plugin for TcnTubeScreamer {
    const NAME: &'static str = "micro-TCN TubeScreamer";
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

        if self.model.is_some() {
            if let Some(m) = self.model.as_mut() {
                m.allocate_block_buffers(buffer_config.max_buffer_size as usize);
            }
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
                "model mismatch: {} — this wrapper only accepts fixed-setting TubeScreamer models",
                e
            );
            return false;
        }

        nih_log!("loaded TCN model from {}", source);
        self.model = Some(model);
        if let Some(m) = self.model.as_mut() {
            m.allocate_block_buffers(buffer_config.max_buffer_size as usize);
        }
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

        // Zero-param conditioning: the gen MLP has no input, so FiLM scale/shift
        // is purely bias-driven. Still cheap to recompute once per buffer.
        model.update_conditioning(&self.cond_scratch);

        for ch in buffer.as_slice() {
            model.process_block_inplace(ch);
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for TcnTubeScreamer {
    const CLAP_ID: &'static str = "com.microtcn.tcn-tubescreamer";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Neural tube-screamer distortion — micro-TCN inference.");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Mono,
        ClapFeature::Distortion,
    ];
}

impl Vst3Plugin for TcnTubeScreamer {
    const VST3_CLASS_ID: [u8; 16] = *b"microtcnTCNTscrV";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Distortion];
}

nih_export_clap!(TcnTubeScreamer);
nih_export_vst3!(TcnTubeScreamer);

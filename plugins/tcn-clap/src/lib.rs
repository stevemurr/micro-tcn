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

pub mod model;
use model::TcnModel;

/// Directory containing the currently loaded plugin binary.
///
/// On Unix we find it via `dladdr` on a symbol from this crate. On Windows we
/// currently fall back to `None` — users there can set `TCN_CLAP_MODEL`.
fn plugin_binary_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    unsafe {
        use std::ffi::CStr;
        use std::os::raw::{c_char, c_int, c_void};

        #[repr(C)]
        struct DlInfo {
            dli_fname: *const c_char,
            dli_fbase: *mut c_void,
            dli_sname: *const c_char,
            dli_saddr: *mut c_void,
        }

        extern "C" {
            fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
        }

        let mut info: DlInfo = std::mem::zeroed();
        if dladdr(plugin_binary_dir as *const c_void, &mut info) == 0 {
            return None;
        }
        if info.dli_fname.is_null() {
            return None;
        }
        let path = CStr::from_ptr(info.dli_fname).to_str().ok()?;
        PathBuf::from(path).parent().map(PathBuf::from)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

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

/// Default model, baked into the binary at compile time. Kept at
/// `plugins/tcn-clap/assets/tcn.json`; replace and rebuild to ship a new
/// default. Runtime overrides (env var or bundle Resources/) still win if set.
const DEFAULT_MODEL_JSON: &str = include_str!("../assets/tcn.json");

enum ModelSource {
    Path(PathBuf),
    Embedded(&'static str),
}

/// Resolve where to load the TCN model from, in priority order:
/// 1. `TCN_CLAP_MODEL` env var (explicit override)
/// 2. macOS bundle: `Contents/Resources/tcn.json`
/// 3. Next to the plugin binary (Linux/Windows single-file `.clap`)
/// 4. Embedded default compiled into the binary
fn locate_model() -> ModelSource {
    if let Ok(p) = env::var("TCN_CLAP_MODEL") {
        return ModelSource::Path(PathBuf::from(p));
    }

    if let Some(binary_dir) = plugin_binary_dir() {
        #[cfg(target_os = "macos")]
        if let Some(contents) = binary_dir.parent() {
            let resources = contents.join("Resources").join("tcn.json");
            if resources.exists() {
                return ModelSource::Path(resources);
            }
        }

        let next_to_plugin = binary_dir.join("tcn.json");
        if next_to_plugin.exists() {
            return ModelSource::Path(next_to_plugin);
        }
    }

    ModelSource::Embedded(DEFAULT_MODEL_JSON)
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

        let result = match locate_model() {
            ModelSource::Path(path) => {
                TcnModel::load_from_json_file(&path).map(|m| (m, format!("file {}", path.display())))
            }
            ModelSource::Embedded(text) => {
                TcnModel::load_from_json_str(text).map(|m| (m, "embedded default".to_string()))
            }
        };

        match result {
            Ok((m, source)) => {
                nih_log!("loaded TCN model from {}", source);
                self.model = Some(m);
                true
            }
            Err(e) => {
                nih_log!("failed to load TCN model: {}", e);
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

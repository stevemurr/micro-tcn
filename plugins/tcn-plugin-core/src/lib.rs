//! Shared runtime for micro-TCN nih-plug wrappers.
//!
//! Wrappers pull in this crate for:
//!   * [`TcnModel`] — the pure-Rust inference engine (no libtorch / RTNeural).
//!   * [`locate_model`] / [`ModelSource`] — the priority-ordered search for a
//!     `tcn.json` at runtime (env var → bundle Resources → next-to-binary →
//!     baked-in embedded default).
//!
//! The wrappers themselves only contribute a `#[derive(Params)]` struct, the
//! per-plugin CLAP_ID / description / class_id, and a mapping from their
//! declared knobs to the flat `&[f32]` the model wants. See
//! `plugins/tcn-la2a/src/lib.rs` for a minimal example.

pub mod model;
pub use model::TcnModel;

use std::env;
use std::path::PathBuf;

/// Where the plugin binary lives on disk. On Unix we resolve this via `dladdr`
/// against a symbol in *this* crate; since the rlib is statically linked into
/// each wrapper's cdylib, the returned path is the wrapper's `.clap` /
/// `.vst3`, not the core lib. On Windows we return `None` — fall back to the
/// env-var override path.
pub fn plugin_binary_dir() -> Option<PathBuf> {
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

/// Where the model JSON should come from.
pub enum ModelSource {
    /// Load from an explicit path on disk.
    Path(PathBuf),
    /// Parse the baked-in string compiled into the wrapper binary.
    Embedded(&'static str),
}

/// Resolve where to load the TCN model from, in priority order:
///
/// 1. `env_override` env var (explicit override, e.g. `TCN_LA2A_MODEL`).
/// 2. macOS bundle: `Contents/Resources/tcn.json`.
/// 3. Next to the plugin binary (Linux/Windows single-file `.clap`).
/// 4. The embedded default compiled into the wrapper.
///
/// `env_override` is per-plugin so two wrappers installed at once don't collide.
pub fn locate_model(env_override: &str, embedded: &'static str) -> ModelSource {
    if let Ok(p) = env::var(env_override) {
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

    ModelSource::Embedded(embedded)
}

/// Convenience: resolve + load, returning the model plus a human-readable
/// source description for logging.
pub fn load_model(
    env_override: &str,
    embedded: &'static str,
) -> Result<(TcnModel, String), String> {
    match locate_model(env_override, embedded) {
        ModelSource::Path(path) => {
            let m = TcnModel::load_from_json_file(&path)?;
            Ok((m, format!("file {}", path.display())))
        }
        ModelSource::Embedded(text) => {
            let m = TcnModel::load_from_json_str(text)?;
            Ok((m, "embedded default".to_string()))
        }
    }
}

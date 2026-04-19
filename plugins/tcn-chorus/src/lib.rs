//! Raw-CLAP TubeScreamer wrapper.
//!
//! Bypasses nih-plug entirely so the audio thread calls our process() with
//! zero framework overhead. Returns CLAP_PROCESS_SLEEP after a block of silent
//! output, which guarantees the host stops calling us at idle (nih-plug had no
//! way to return SLEEP — its Tail variant mapped to CONTINUE instead).
//!
//! Internal 128-sample block buffering keeps SGEMM amortization high regardless
//! of the host's buffer size.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicU32, Ordering};

use clap_sys::entry::clap_plugin_entry;
use clap_sys::ext::audio_ports::{
    clap_audio_port_info, clap_plugin_audio_ports, CLAP_AUDIO_PORT_IS_MAIN, CLAP_EXT_AUDIO_PORTS,
    CLAP_PORT_MONO,
};
use clap_sys::ext::latency::{clap_plugin_latency, CLAP_EXT_LATENCY};
use clap_sys::factory::plugin_factory::{clap_plugin_factory, CLAP_PLUGIN_FACTORY_ID};
use clap_sys::host::clap_host;
use clap_sys::id::CLAP_INVALID_ID;
use clap_sys::plugin::{clap_plugin, clap_plugin_descriptor};
use clap_sys::process::{
    clap_process, clap_process_status, CLAP_PROCESS_CONTINUE_IF_NOT_QUIET, CLAP_PROCESS_ERROR,
    CLAP_PROCESS_SLEEP,
};
use clap_sys::string_sizes::CLAP_NAME_SIZE;
use clap_sys::version::CLAP_VERSION;

use tcn_plugin_core::TcnModel;

// ─── Constants ────────────────────────────────────────────────────────────────

const INTERNAL_N: usize = 128;
const DEFAULT_MODEL: &str = include_str!("../assets/tcn.json");

// ─── Plugin inner state (audio thread) ────────────────────────────────────────

struct Inner {
    model: Option<TcnModel>,
    input_accum: Vec<f32>,
    output_accum: Vec<f32>,
    accum_fill: usize,
    out_drain_pos: usize,
    /// Set after a block of all-zero output — tells process() to return SLEEP.
    last_silent: bool,
}

impl Inner {
    fn new() -> Self {
        Self {
            model: None,
            input_accum: vec![0.0; INTERNAL_N],
            output_accum: vec![0.0; INTERNAL_N],
            accum_fill: 0,
            out_drain_pos: 0,
            last_silent: false,
        }
    }

    fn reset(&mut self) {
        if let Some(m) = self.model.as_mut() { m.reset(); }
        self.input_accum.fill(0.0);
        self.output_accum.fill(0.0);
        self.accum_fill = 0;
        self.out_drain_pos = 0;
        self.last_silent = false;
    }
}

// ─── Instance ─────────────────────────────────────────────────────────────────

/// CLAP plugin instance. `clap` MUST be the first field so a *const clap_plugin
/// can be cast to *mut Instance.
#[repr(C)]
struct Instance {
    clap: clap_plugin,
    _host: *const clap_host,
    inner: std::cell::UnsafeCell<Inner>,
    // Atomic flag written by main thread, read by audio thread.
    is_active: AtomicU32,
}

// Raw-pointer fields: we manage lifetime manually via Box::into_raw / from_raw.
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

#[inline(always)]
unsafe fn get_inner(plugin: *const clap_plugin) -> &'static mut Inner {
    let inst = &*(plugin as *const Instance);
    &mut *inst.inner.get()
}

// ─── CLAP callbacks ───────────────────────────────────────────────────────────

unsafe extern "C" fn plugin_init(_plugin: *const clap_plugin) -> bool { true }

unsafe extern "C" fn plugin_destroy(plugin: *const clap_plugin) {
    drop(Box::from_raw(plugin as *mut Instance));
}

unsafe extern "C" fn plugin_activate(
    plugin: *const clap_plugin,
    _sample_rate: f64,
    _min_frames: u32,
    _max_frames: u32,
) -> bool {
    let inner = get_inner(plugin);
    inner.reset();

    if inner.model.is_none() {
        match TcnModel::load_from_json_str(DEFAULT_MODEL) {
            Ok(mut m) => {
                m.require_nparams(0).ok();
                m.allocate_block_buffers(INTERNAL_N);
                m.update_conditioning(&[]);
                inner.model = Some(m);
            }
            Err(_) => return false,
        }
    } else if let Some(m) = inner.model.as_mut() {
        m.allocate_block_buffers(INTERNAL_N);
        m.update_conditioning(&[]);
    }

    (*(&*(plugin as *const Instance) as *const Instance as *mut Instance))
        .is_active
        .store(1, Ordering::SeqCst);
    true
}

unsafe extern "C" fn plugin_deactivate(plugin: *const clap_plugin) {
    (*(&*(plugin as *const Instance) as *const Instance as *mut Instance))
        .is_active
        .store(0, Ordering::SeqCst);
}

unsafe extern "C" fn plugin_start_processing(_plugin: *const clap_plugin) -> bool { true }
unsafe extern "C" fn plugin_stop_processing(_plugin: *const clap_plugin) {}

unsafe extern "C" fn plugin_reset(plugin: *const clap_plugin) {
    get_inner(plugin).reset();
}

unsafe extern "C" fn plugin_process(
    plugin: *const clap_plugin,
    process: *const clap_process,
) -> clap_process_status {
    let proc = &*process;
    let inner = get_inner(plugin);

    let model = match inner.model.as_mut() {
        Some(m) => m,
        None => return CLAP_PROCESS_ERROR,
    };

    let n = proc.frames_count as usize;
    if n == 0 { return CLAP_PROCESS_CONTINUE_IF_NOT_QUIET; }

    if proc.audio_inputs_count == 0 || proc.audio_outputs_count == 0 {
        return CLAP_PROCESS_SLEEP;
    }

    let in_buf = &*proc.audio_inputs;
    let out_buf = &mut *proc.audio_outputs;

    if in_buf.data32.is_null() || out_buf.data32.is_null() {
        return CLAP_PROCESS_SLEEP;
    }

    let in_ch = std::slice::from_raw_parts(*in_buf.data32, n);
    let out_ch = std::slice::from_raw_parts_mut(*out_buf.data32, n);

    let mut all_silent = true;

    for host_t in 0..n {
        let in_s = in_ch[host_t];

        let out_s = inner.output_accum[inner.out_drain_pos];
        inner.out_drain_pos += 1;

        inner.input_accum[inner.accum_fill] = in_s;
        inner.accum_fill += 1;

        if inner.accum_fill == INTERNAL_N {
            let block = &mut inner.input_accum[..INTERNAL_N];
            let peak = block.iter().copied().fold(0.0f32, |m, s| m.max(s.abs()));
            if peak >= 1.5e-5 {
                model.process_block_inplace(block);
            } else {
                block.fill(0.0);
            }
            inner.output_accum.copy_from_slice(block);
            inner.accum_fill = 0;
            inner.out_drain_pos = 0;
        }

        if out_s.abs() >= 1.5e-5 { all_silent = false; }
        out_ch[host_t] = out_s;
    }

    if all_silent {
        // Set silence flag so hosts that check constant_mask also see quiet.
        out_buf.constant_mask = 1;
        inner.last_silent = true;
        CLAP_PROCESS_SLEEP
    } else {
        out_buf.constant_mask = 0;
        inner.last_silent = false;
        CLAP_PROCESS_CONTINUE_IF_NOT_QUIET
    }
}

unsafe extern "C" fn plugin_get_extension(
    _plugin: *const clap_plugin,
    id: *const c_char,
) -> *const c_void {
    let id = CStr::from_ptr(id);
    if id == CLAP_EXT_AUDIO_PORTS {
        return &AUDIO_PORTS_EXT as *const _ as *const c_void;
    }
    if id == CLAP_EXT_LATENCY {
        return &LATENCY_EXT as *const _ as *const c_void;
    }
    std::ptr::null()
}

unsafe extern "C" fn plugin_on_main_thread(_plugin: *const clap_plugin) {}

// ─── Audio ports extension ────────────────────────────────────────────────────

unsafe extern "C" fn audio_ports_count(_plugin: *const clap_plugin, _is_input: bool) -> u32 { 1 }

unsafe extern "C" fn audio_ports_get(
    _plugin: *const clap_plugin,
    index: u32,
    _is_input: bool,
    info: *mut clap_audio_port_info,
) -> bool {
    if index != 0 { return false; }
    let info = &mut *info;
    info.id = 0;
    fill_name(&mut info.name, b"Main\0");
    info.flags = CLAP_AUDIO_PORT_IS_MAIN;
    info.channel_count = 1;
    info.port_type = CLAP_PORT_MONO.as_ptr();
    info.in_place_pair = CLAP_INVALID_ID;
    true
}

static AUDIO_PORTS_EXT: clap_plugin_audio_ports = clap_plugin_audio_ports {
    count: Some(audio_ports_count),
    get: Some(audio_ports_get),
};

// ─── Latency extension ────────────────────────────────────────────────────────

unsafe extern "C" fn latency_get(_plugin: *const clap_plugin) -> u32 { INTERNAL_N as u32 }

static LATENCY_EXT: clap_plugin_latency = clap_plugin_latency { get: Some(latency_get) };

// ─── Descriptor ───────────────────────────────────────────────────────────────

struct FeaturesPtr([*const c_char; 4]);
unsafe impl Sync for FeaturesPtr {}
static FEATURES: FeaturesPtr = FeaturesPtr([
    b"audio-effect\0".as_ptr() as *const c_char,
    b"mono\0".as_ptr() as *const c_char,
    b"chorus\0".as_ptr() as *const c_char,
    std::ptr::null(),
]);

static DESCRIPTOR: clap_plugin_descriptor = clap_plugin_descriptor {
    clap_version: CLAP_VERSION,
    id: b"com.microtcn.tcn-chorus\0".as_ptr() as *const c_char,
    name: b"micro-TCN Chorus\0".as_ptr() as *const c_char,
    vendor: b"micro-tcn\0".as_ptr() as *const c_char,
    url: b"https://github.com/stevemurr/micro-tcn\0".as_ptr() as *const c_char,
    manual_url: b"\0".as_ptr() as *const c_char,
    support_url: b"\0".as_ptr() as *const c_char,
    version: b"0.1.0\0".as_ptr() as *const c_char,
    description: b"Neural Boss CE-3 chorus.\0".as_ptr() as *const c_char,
    features: FEATURES.0.as_ptr(),
};

// ─── Factory ──────────────────────────────────────────────────────────────────

unsafe extern "C" fn factory_count(_factory: *const clap_plugin_factory) -> u32 { 1 }

unsafe extern "C" fn factory_get_descriptor(
    _factory: *const clap_plugin_factory,
    index: u32,
) -> *const clap_plugin_descriptor {
    if index == 0 { &DESCRIPTOR } else { std::ptr::null() }
}

unsafe extern "C" fn factory_create(
    _factory: *const clap_plugin_factory,
    host: *const clap_host,
    plugin_id: *const c_char,
) -> *const clap_plugin {
    let id = CStr::from_ptr(plugin_id);
    if id != CStr::from_ptr(DESCRIPTOR.id) {
        return std::ptr::null();
    }
    let inst = Box::new(Instance {
        clap: clap_plugin {
            desc: &DESCRIPTOR,
            plugin_data: std::ptr::null_mut(), // set below
            init: Some(plugin_init),
            destroy: Some(plugin_destroy),
            activate: Some(plugin_activate),
            deactivate: Some(plugin_deactivate),
            start_processing: Some(plugin_start_processing),
            stop_processing: Some(plugin_stop_processing),
            reset: Some(plugin_reset),
            process: Some(plugin_process),
            get_extension: Some(plugin_get_extension),
            on_main_thread: Some(plugin_on_main_thread),
        },
        _host: host,
        inner: std::cell::UnsafeCell::new(Inner::new()),
        is_active: AtomicU32::new(0),
    });
    let raw = Box::into_raw(inst);
    (*raw).clap.plugin_data = raw as *mut c_void;
    &(*raw).clap
}

static FACTORY: clap_plugin_factory = clap_plugin_factory {
    get_plugin_count: Some(factory_count),
    get_plugin_descriptor: Some(factory_get_descriptor),
    create_plugin: Some(factory_create),
};

// ─── Entry point ──────────────────────────────────────────────────────────────

unsafe extern "C" fn entry_init(_path: *const c_char) -> bool { true }
unsafe extern "C" fn entry_deinit() {}
unsafe extern "C" fn entry_get_factory(factory_id: *const c_char) -> *const c_void {
    let id = CStr::from_ptr(factory_id);
    if id == CLAP_PLUGIN_FACTORY_ID {
        return &FACTORY as *const _ as *const c_void;
    }
    std::ptr::null()
}

#[no_mangle]
pub static clap_entry: clap_plugin_entry = clap_plugin_entry {
    clap_version: CLAP_VERSION,
    init: Some(entry_init),
    deinit: Some(entry_deinit),
    get_factory: Some(entry_get_factory),
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn fill_name(dst: &mut [c_char; CLAP_NAME_SIZE], src: &[u8]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = *s as c_char;
    }
}

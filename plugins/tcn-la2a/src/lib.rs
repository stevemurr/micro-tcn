//! Raw-CLAP LA2A wrapper.
//!
//! Same structural fix as tcn-tubescreamer: bypasses nih-plug so we can
//! return CLAP_PROCESS_SLEEP and set constant_mask on silence — two things
//! nih-plug cannot do — guaranteeing the host stops calling at idle.
//!
//! Two CLAP parameters are exposed: peak_reduction (0.0–1.0) and limit
//! (0.0 = compress, 1.0 = limit). GUI is intentionally omitted for now;
//! DAW automation and knobs on the plugin strip are the primary interface.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicU32, Ordering};

use clap_sys::entry::clap_plugin_entry;
use clap_sys::events::{
    clap_event_header, clap_event_param_value, CLAP_CORE_EVENT_SPACE_ID,
    CLAP_EVENT_PARAM_VALUE,
};
use clap_sys::ext::audio_ports::{
    clap_audio_port_info, clap_plugin_audio_ports, CLAP_AUDIO_PORT_IS_MAIN, CLAP_EXT_AUDIO_PORTS,
    CLAP_PORT_MONO,
};
use clap_sys::ext::latency::{clap_plugin_latency, CLAP_EXT_LATENCY};
use clap_sys::ext::params::{
    clap_param_info, clap_plugin_params, CLAP_EXT_PARAMS, CLAP_PARAM_IS_AUTOMATABLE,
    CLAP_PARAM_IS_STEPPED,
};
use clap_sys::ext::state::{clap_plugin_state, CLAP_EXT_STATE};
use clap_sys::stream::{clap_istream, clap_ostream};
use clap_sys::factory::plugin_factory::{clap_plugin_factory, CLAP_PLUGIN_FACTORY_ID};
use clap_sys::host::clap_host;
use clap_sys::id::{clap_id, CLAP_INVALID_ID};
use clap_sys::plugin::{clap_plugin, clap_plugin_descriptor};
use clap_sys::process::{
    clap_process, clap_process_status, CLAP_PROCESS_CONTINUE_IF_NOT_QUIET, CLAP_PROCESS_ERROR,
    CLAP_PROCESS_SLEEP,
};
use clap_sys::string_sizes::{CLAP_NAME_SIZE, CLAP_PATH_SIZE};
use clap_sys::version::CLAP_VERSION;

use tcn_plugin_core::TcnModel;

// ─── Constants ────────────────────────────────────────────────────────────────

const INTERNAL_N: usize = 128;
const NPARAMS: usize = 2;
const DEFAULT_MODEL: &str = include_str!("../assets/tcn.json");

const PARAM_PEAK_RED: clap_id = 0;
const PARAM_LIMIT:    clap_id = 1;

// ─── Plugin inner state (audio thread) ────────────────────────────────────────

struct Inner {
    model: Option<TcnModel>,
    input_accum: Vec<f32>,
    output_accum: Vec<f32>,
    accum_fill: usize,
    out_drain_pos: usize,
    cond_scratch: [f32; NPARAMS],
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
            cond_scratch: [0.0; NPARAMS],
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

/// `clap` must be first field for pointer casting.
#[repr(C)]
struct Instance {
    clap: clap_plugin,
    _host: *const clap_host,
    inner: std::cell::UnsafeCell<Inner>,
    /// Params stored as f32 bits for lock-free cross-thread reads.
    /// [0] = peak_reduction (0.0–1.0), [1] = limit (0.0 or 1.0)
    params: [AtomicU32; NPARAMS],
}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

#[inline(always)]
unsafe fn get_inner(plugin: *const clap_plugin) -> &'static mut Inner {
    &mut *(*( plugin as *const Instance)).inner.get()
}

#[inline(always)]
unsafe fn get_instance(plugin: *const clap_plugin) -> &'static Instance {
    &*(plugin as *const Instance)
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

    // Load current param values into cond_scratch.
    let inst = get_instance(plugin);
    inner.cond_scratch[0] = f32::from_bits(inst.params[0].load(Ordering::Relaxed)); // limit
    inner.cond_scratch[1] = f32::from_bits(inst.params[1].load(Ordering::Relaxed)); // peak_red

    if inner.model.is_none() {
        match TcnModel::load_from_json_str(DEFAULT_MODEL) {
            Ok(mut m) => {
                if m.require_nparams(NPARAMS).is_err() { return false; }
                m.allocate_block_buffers(INTERNAL_N);
                m.update_conditioning(&inner.cond_scratch);
                inner.model = Some(m);
            }
            Err(_) => return false,
        }
    } else if let Some(m) = inner.model.as_mut() {
        m.allocate_block_buffers(INTERNAL_N);
        m.update_conditioning(&inner.cond_scratch);
    }
    true
}

unsafe extern "C" fn plugin_deactivate(_plugin: *const clap_plugin) {}
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

    // Drain param events from the host's input event list.
    let inst = get_instance(plugin);
    if !proc.in_events.is_null() {
        let events = &*proc.in_events;
        if let Some(size_fn) = events.size {
            let count = size_fn(proc.in_events);
            if let Some(get_fn) = events.get {
                for i in 0..count {
                    let ev = get_fn(proc.in_events, i);
                    if ev.is_null() { continue; }
                    let hdr = &*(ev as *const clap_event_header);
                    if hdr.space_id == CLAP_CORE_EVENT_SPACE_ID
                        && hdr.type_ == CLAP_EVENT_PARAM_VALUE
                    {
                        let pev = &*(ev as *const clap_event_param_value);
                        match pev.param_id {
                            PARAM_PEAK_RED => {
                                inst.params[1].store((pev.value as f32).to_bits(), Ordering::Relaxed);
                            }
                            PARAM_LIMIT => {
                                inst.params[0].store((pev.value as f32).to_bits(), Ordering::Relaxed);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    // Rebuild conditioning if params changed.
    let limit    = f32::from_bits(inst.params[0].load(Ordering::Relaxed));
    let peak_red = f32::from_bits(inst.params[1].load(Ordering::Relaxed));
    if limit != inner.cond_scratch[0] || peak_red != inner.cond_scratch[1] {
        inner.cond_scratch[0] = limit;
        inner.cond_scratch[1] = peak_red;
        model.update_conditioning(&inner.cond_scratch);
    }

    let in_ch  = std::slice::from_raw_parts(*in_buf.data32, n);
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
    if id == CLAP_EXT_AUDIO_PORTS { return &AUDIO_PORTS_EXT as *const _ as *const c_void; }
    if id == CLAP_EXT_LATENCY     { return &LATENCY_EXT     as *const _ as *const c_void; }
    if id == CLAP_EXT_PARAMS      { return &PARAMS_EXT      as *const _ as *const c_void; }
    if id == CLAP_EXT_STATE       { return &STATE_EXT       as *const _ as *const c_void; }
    std::ptr::null()
}

unsafe extern "C" fn plugin_on_main_thread(_plugin: *const clap_plugin) {}

// ─── Audio ports ──────────────────────────────────────────────────────────────

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

// ─── Latency ──────────────────────────────────────────────────────────────────

unsafe extern "C" fn latency_get(_plugin: *const clap_plugin) -> u32 { INTERNAL_N as u32 }

static LATENCY_EXT: clap_plugin_latency = clap_plugin_latency { get: Some(latency_get) };

// ─── Params ───────────────────────────────────────────────────────────────────

unsafe extern "C" fn params_count(_plugin: *const clap_plugin) -> u32 { NPARAMS as u32 }

unsafe extern "C" fn params_get_info(
    _plugin: *const clap_plugin,
    index: u32,
    info: *mut clap_param_info,
) -> bool {
    let info = &mut *info;
    info.cookie = std::ptr::null_mut();
    info.module = [0; CLAP_PATH_SIZE];
    match index {
        0 => {
            info.id = PARAM_PEAK_RED;
            info.flags = CLAP_PARAM_IS_AUTOMATABLE;
            fill_name(&mut info.name, b"Peak Reduction\0");
            info.min_value = 0.0;
            info.max_value = 1.0;
            info.default_value = 0.5;
        }
        1 => {
            info.id = PARAM_LIMIT;
            info.flags = CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_STEPPED;
            fill_name(&mut info.name, b"Limit\0");
            info.min_value = 0.0;
            info.max_value = 1.0;
            info.default_value = 0.0;
        }
        _ => return false,
    }
    true
}

unsafe extern "C" fn params_get_value(
    plugin: *const clap_plugin,
    param_id: clap_id,
    out_value: *mut f64,
) -> bool {
    let inst = get_instance(plugin);
    let v = match param_id {
        PARAM_PEAK_RED => f32::from_bits(inst.params[1].load(Ordering::Relaxed)) as f64,
        PARAM_LIMIT    => f32::from_bits(inst.params[0].load(Ordering::Relaxed)) as f64,
        _ => return false,
    };
    *out_value = v;
    true
}

unsafe extern "C" fn params_value_to_text(
    _plugin: *const clap_plugin,
    param_id: clap_id,
    value: f64,
    out_buf: *mut c_char,
    capacity: u32,
) -> bool {
    if out_buf.is_null() || capacity == 0 { return false; }
    let s = match param_id {
        PARAM_PEAK_RED => format!("{:.2}", value),
        PARAM_LIMIT    => if value >= 0.5 { "Limit".to_string() } else { "Compress".to_string() },
        _ => return false,
    };
    let bytes = s.as_bytes();
    let write = bytes.len().min(capacity as usize - 1);
    for (i, b) in bytes[..write].iter().enumerate() {
        *out_buf.add(i) = *b as c_char;
    }
    *out_buf.add(write) = 0;
    true
}

unsafe extern "C" fn params_text_to_value(
    _plugin: *const clap_plugin,
    param_id: clap_id,
    text: *const c_char,
    out_value: *mut f64,
) -> bool {
    if text.is_null() { return false; }
    let s = CStr::from_ptr(text).to_string_lossy();
    let v: f64 = match param_id {
        PARAM_PEAK_RED => s.parse().unwrap_or(0.5),
        PARAM_LIMIT => if s.to_lowercase().contains("limit") { 1.0 } else { 0.0 },
        _ => return false,
    };
    *out_value = v;
    true
}

unsafe extern "C" fn params_flush(
    plugin: *const clap_plugin,
    in_events: *const clap_sys::events::clap_input_events,
    _out_events: *const clap_sys::events::clap_output_events,
) {
    if in_events.is_null() { return; }
    let events = &*in_events;
    let inst = get_instance(plugin);
    if let Some(size_fn) = events.size {
        let count = size_fn(in_events);
        if let Some(get_fn) = events.get {
            for i in 0..count {
                let ev = get_fn(in_events, i);
                if ev.is_null() { continue; }
                let hdr = &*(ev as *const clap_event_header);
                if hdr.space_id == CLAP_CORE_EVENT_SPACE_ID
                    && hdr.type_ == CLAP_EVENT_PARAM_VALUE
                {
                    let pev = &*(ev as *const clap_event_param_value);
                    match pev.param_id {
                        PARAM_PEAK_RED => {
                            inst.params[1].store((pev.value as f32).to_bits(), Ordering::Relaxed);
                        }
                        PARAM_LIMIT => {
                            inst.params[0].store((pev.value as f32).to_bits(), Ordering::Relaxed);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

static PARAMS_EXT: clap_plugin_params = clap_plugin_params {
    count: Some(params_count),
    get_info: Some(params_get_info),
    get_value: Some(params_get_value),
    value_to_text: Some(params_value_to_text),
    text_to_value: Some(params_text_to_value),
    flush: Some(params_flush),
};

// ─── State (save / load) ──────────────────────────────────────────────────────
// Format: 8 bytes — [limit: f32 LE][peak_reduction: f32 LE]

unsafe extern "C" fn state_save(plugin: *const clap_plugin, stream: *const clap_ostream) -> bool {
    let inst = get_instance(plugin);
    let limit    = f32::from_bits(inst.params[0].load(Ordering::Relaxed));
    let peak_red = f32::from_bits(inst.params[1].load(Ordering::Relaxed));
    let bytes: [u8; 8] = {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&limit.to_le_bytes());
        b[4..].copy_from_slice(&peak_red.to_le_bytes());
        b
    };
    let write = (*stream).write.expect("null ostream write");
    let written = write(stream, bytes.as_ptr() as *const c_void, 8);
    written == 8
}

unsafe extern "C" fn state_load(plugin: *const clap_plugin, stream: *const clap_istream) -> bool {
    let mut bytes = [0u8; 8];
    let read = (*stream).read.expect("null istream read");
    let n = read(stream, bytes.as_mut_ptr() as *mut c_void, 8);
    if n != 8 { return false; }
    let limit    = f32::from_le_bytes(bytes[..4].try_into().unwrap());
    let peak_red = f32::from_le_bytes(bytes[4..].try_into().unwrap());
    let inst = get_instance(plugin);
    inst.params[0].store(limit.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    inst.params[1].store(peak_red.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    true
}

static STATE_EXT: clap_plugin_state = clap_plugin_state {
    save: Some(state_save),
    load: Some(state_load),
};

// ─── Descriptor ───────────────────────────────────────────────────────────────

struct FeaturesPtr([*const c_char; 4]);
unsafe impl Sync for FeaturesPtr {}
static FEATURES: FeaturesPtr = FeaturesPtr([
    b"audio-effect\0".as_ptr() as *const c_char,
    b"mono\0".as_ptr() as *const c_char,
    b"compressor\0".as_ptr() as *const c_char,
    std::ptr::null(),
]);

static DESCRIPTOR: clap_plugin_descriptor = clap_plugin_descriptor {
    clap_version: CLAP_VERSION,
    id: b"com.microtcn.tcn-la2a\0".as_ptr() as *const c_char,
    name: b"micro-TCN LA2A\0".as_ptr() as *const c_char,
    vendor: b"micro-tcn\0".as_ptr() as *const c_char,
    url: b"https://github.com/stevemurr/micro-tcn\0".as_ptr() as *const c_char,
    manual_url: b"\0".as_ptr() as *const c_char,
    support_url: b"\0".as_ptr() as *const c_char,
    version: b"0.1.0\0".as_ptr() as *const c_char,
    description: b"Neural dynamic range compressor modeled on LA2A.\0".as_ptr() as *const c_char,
    features: FEATURES.0.as_ptr(),
};

// ─── Factory ──────────────────────────────────────────────────────────────────

unsafe extern "C" fn factory_count(_: *const clap_plugin_factory) -> u32 { 1 }

unsafe extern "C" fn factory_get_descriptor(
    _: *const clap_plugin_factory,
    index: u32,
) -> *const clap_plugin_descriptor {
    if index == 0 { &DESCRIPTOR } else { std::ptr::null() }
}

unsafe extern "C" fn factory_create(
    _: *const clap_plugin_factory,
    host: *const clap_host,
    plugin_id: *const c_char,
) -> *const clap_plugin {
    let id = CStr::from_ptr(plugin_id);
    if id != CStr::from_ptr(DESCRIPTOR.id) { return std::ptr::null(); }
    let inst = Box::new(Instance {
        clap: clap_plugin {
            desc: &DESCRIPTOR,
            plugin_data: std::ptr::null_mut(),
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
        params: [
            AtomicU32::new(0.0f32.to_bits()), // limit = 0.0
            AtomicU32::new(0.5f32.to_bits()), // peak_reduction = 0.5
        ],
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

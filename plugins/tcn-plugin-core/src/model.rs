//! TCN inference runtime — block-level SGEMM via Apple Accelerate.
//!
//! Each TCN block's dilated conv is implemented as:
//!   1. Gather: ring buffer → contiguous [N, K×C_in] matrix
//!   2. SGEMM:  [N, K×C_in] × [K×C_in, C_out] → [N, C_out]
//!
//! Ring layout is [ring_len, C_in] (time-major) so each tap gather is a
//! single contiguous copy_from_slice instead of C_in scattered reads.
//!
//! Weights are stored (K, C_in, C_out) = [K×C_in, C_out] to match the
//! gather layout for a no-transpose SGEMM.
//!
//! Call `allocate_block_buffers(max_frames)` once after loading (done
//! automatically in load_from_json_str with a 512-sample default, and
//! overridden by the CLAP activate callback with the host's actual max).

use serde::Deserialize;
use std::path::Path;

// ─── JSON schema (matches microtcn/export.py) ─────────────────────────────────

#[derive(Deserialize)]
struct ModelJson {
    version: u32,
    arch: String,
    config: ConfigJson,
    gen: Vec<LinearJson>,
    blocks: Vec<BlockJson>,
    output: Conv1dJson,
}

#[derive(Deserialize, Clone)]
struct ConfigJson {
    #[allow(dead_code)]
    nblocks: usize,
    #[allow(dead_code)]
    kernel_size: usize,
    #[allow(dead_code)]
    dilation_growth: usize,
    channel_width: usize,
    #[allow(dead_code)]
    causal: bool,
    nparams: usize,
    sample_rate: u32,
    receptive_field: usize,
}

#[derive(Deserialize)]
struct LinearJson {
    #[allow(dead_code)]
    in_features: usize,
    out_features: usize,
    weight: Vec<Vec<f32>>, // (out, in)
    bias: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct Conv1dJson {
    in_channels: usize,
    #[allow(dead_code)]
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
    weight: Vec<Vec<Vec<f32>>>, // (out, in/groups, kernel)
    bias: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct Bn1dJson {
    #[allow(dead_code)]
    num_features: usize,
    #[allow(dead_code)]
    affine: bool,
    eps: f32,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
}

#[derive(Deserialize)]
struct PreluJson {
    #[allow(dead_code)]
    num_parameters: usize,
    weight: Vec<f32>,
}

#[derive(Deserialize)]
struct BlockJson {
    conv1: Conv1dJson,
    bn: Bn1dJson,
    adaptor: LinearJson,
    prelu: PreluJson,
    res: Conv1dJson,
    #[allow(dead_code)]
    causal: bool,
}

// ─── Flat runtime representation ──────────────────────────────────────────────

struct Linear {
    weight: Vec<f32>, // row-major (out, in)
    bias: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    fn from_json(l: LinearJson) -> Self {
        let mut weight = Vec::with_capacity(l.out_features * l.in_features);
        for row in &l.weight {
            weight.extend_from_slice(row);
        }
        let bias = l.bias.unwrap_or_else(|| vec![0.0; l.out_features]);
        Self { weight, bias, in_features: l.in_features, out_features: l.out_features }
    }

    #[inline(always)]
    fn forward(&self, x: &[f32], out: &mut [f32]) {
        for o in 0..self.out_features {
            let row = &self.weight[o * self.in_features..(o + 1) * self.in_features];
            let mut acc = self.bias[o];
            for i in 0..self.in_features {
                acc += row[i] * x[i];
            }
            out[o] = acc;
        }
    }
}

struct TcnBlock {
    // conv1 weights in (K, C_in, C_out) order: index = (kk * in_ch + i) * out_ch + o
    // Laid out as [K*C_in, C_out] for a no-transpose SGEMM against gather[N, K*C_in].
    conv1_weight: Vec<f32>,
    conv1_in_channels: usize,
    conv1_out_channels: usize,
    conv1_kernel: usize,
    // Lag (in samples) for each kernel tap: lag[kk] = (K-1-kk) * D
    conv1_lags: Vec<usize>,

    bn_mean: Vec<f32>,
    bn_var_rsqrt: Vec<f32>,

    adaptor: Linear,

    prelu_slope: Vec<f32>,

    // Residual: dense stored as [C_in, C_out] (transposed from JSON's [out, in])
    // so the residual SGEMM is also no-transpose: [N, C_in] × [C_in, C_out].
    // Depthwise: per-output-channel scalars [C_out].
    res_dense: Option<Vec<f32>>,
    res_depthwise: Option<Vec<f32>>,

    // FiLM affine, refreshed on each param change via update_conditioning.
    film_scale: Vec<f32>,
    film_shift: Vec<f32>,

    // Ring buffer: [ring_len, C_in] (time-major).
    // Tap k for all C_in channels at position idx is a contiguous C_in-element slice.
    ring: Vec<f32>,
    ring_len: usize,
    ring_pos: usize,

    // PyTorch's causal_crop shifts the residual by one sample. We keep the last
    // sample of the block's input here so the first sample of the next buffer
    // can use it as its t=-1 residual input.
    prev_input: Vec<f32>,
}

pub struct TcnModel {
    config: ConfigJson,

    gen: Vec<Linear>,
    blocks: Vec<TcnBlock>,

    // Output 1×1 conv: (1, C, 1) collapsed to a C-length weight + scalar bias.
    output_weight: Vec<f32>,
    output_bias: f32,

    // Pre-allocated processing buffers. Sized for max_block_size samples.
    // Resized by allocate_block_buffers(); default 512 set on load.
    max_block_size: usize,
    max_k_cin: usize,      // max K*C_in across all blocks (determines gather_buf width)
    gather_buf: Vec<f32>,  // [max_N × max_k_cin]
    buf_a: Vec<f32>,       // [max_N × channel_width] — ping-pong pair
    buf_b: Vec<f32>,       // [max_N × channel_width]
    res_buf: Vec<f32>,     // [max_N × channel_width] — shifted input for residual

    // Scratch for gen MLP + per-block FiLM conditioning.
    gen_scratch_a: Vec<f32>,
    gen_scratch_b: Vec<f32>,
    adaptor_scratch: Vec<f32>,
}

impl TcnModel {
    pub fn load_from_json_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::load_from_json_str(&text)
    }

    pub fn load_from_json_str(text: &str) -> Result<Self, String> {
        let parsed: ModelJson = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if parsed.version != 1 {
            return Err(format!("unsupported export version {}", parsed.version));
        }
        if parsed.arch != "direct" {
            return Err(format!("only arch='direct' is supported, got {}", parsed.arch));
        }

        let channel_width = parsed.config.channel_width;

        let gen = parsed.gen.into_iter().map(Linear::from_json).collect();

        let mut max_k_cin = 0usize;

        let blocks: Vec<TcnBlock> = parsed
            .blocks
            .into_iter()
            .map(|b| {
                let k = b.conv1.kernel_size;
                let d = b.conv1.dilation;
                let in_ch = b.conv1.in_channels;
                let out_ch = b.conv1.out_channels;
                let ring_len = (k - 1) * d + 1;

                // conv1 weights in (K, C_in, C_out) order.
                // JSON gives (out, in/groups, kernel) = b.conv1.weight[o][i][kk].
                let mut conv1_weight = vec![0.0_f32; k * in_ch * out_ch];
                for o in 0..out_ch {
                    for i in 0..in_ch {
                        for kk in 0..k {
                            let dst = (kk * in_ch + i) * out_ch + o;
                            conv1_weight[dst] = b.conv1.weight[o][i][kk];
                        }
                    }
                }

                let conv1_lags: Vec<usize> = (0..k).map(|kk| (k - 1 - kk) * d).collect();

                max_k_cin = max_k_cin.max(k * in_ch);

                let bn_var_rsqrt: Vec<f32> = b
                    .bn
                    .running_var
                    .iter()
                    .map(|v| 1.0 / (v + b.bn.eps).sqrt())
                    .collect();
                let num_features = b.bn.running_mean.len();

                // Residual weights. Dense: stored transposed as [C_in, C_out] so the
                // residual SGEMM ([N, C_in] × [C_in, C_out]) needs no transpose flag.
                let (res_dense, res_depthwise) = if b.res.groups == 1 {
                    // JSON: [out, in, 1]. We want [in, out] = [C_in, C_out].
                    let c_in = b.res.in_channels;
                    let c_out = b.res.out_channels;
                    let mut w = vec![0.0_f32; c_in * c_out];
                    for o in 0..c_out {
                        for i in 0..c_in {
                            w[i * c_out + o] = b.res.weight[o][i][0];
                        }
                    }
                    (Some(w), None)
                } else {
                    // Depthwise: groups == in_ch == out_ch. Collapse to per-channel scalars.
                    let w: Vec<f32> = b.res.weight.iter().map(|m| m[0][0]).collect();
                    (None, Some(w))
                };

                // Ring in [ring_len, C_in] (time-major).
                TcnBlock {
                    conv1_weight,
                    conv1_in_channels: in_ch,
                    conv1_out_channels: out_ch,
                    conv1_kernel: k,
                    conv1_lags,
                    bn_mean: b.bn.running_mean,
                    bn_var_rsqrt,
                    adaptor: Linear::from_json(b.adaptor),
                    prelu_slope: b.prelu.weight,
                    res_dense,
                    res_depthwise,
                    film_scale: vec![1.0; num_features],
                    film_shift: vec![0.0; num_features],
                    ring: vec![0.0; ring_len * in_ch],
                    ring_len,
                    ring_pos: 0,
                    prev_input: vec![0.0; in_ch],
                }
            })
            .collect();

        let output_weight: Vec<f32> = parsed.output.weight[0].iter().map(|row| row[0]).collect();
        let output_bias = parsed.output.bias.and_then(|v| v.first().copied()).unwrap_or(0.0);

        // gen scratch must hold the widest layer IO — inputs (nparams),
        // hidden (16), hidden (32), output (32). For nparams = 0..32 the
        // cap is 32; guard against future models with nparams > 32.
        let gen_scratch_width = parsed.config.nparams.max(32);

        let mut model = Self {
            config: parsed.config,
            gen,
            blocks,
            output_weight,
            output_bias,
            max_block_size: 0,
            max_k_cin,
            gather_buf: Vec::new(),
            buf_a: Vec::new(),
            buf_b: Vec::new(),
            res_buf: Vec::new(),
            gen_scratch_a: vec![0.0; gen_scratch_width],
            gen_scratch_b: vec![0.0; gen_scratch_width],
            adaptor_scratch: vec![0.0; channel_width * 2],
        };
        // Default allocation for bench / offline use.
        model.allocate_block_buffers(512);
        Ok(model)
    }

    /// Pre-allocate processing buffers for up to `max_frames` samples per block.
    /// Call this from the CLAP `activate` callback with the host's reported maximum.
    pub fn allocate_block_buffers(&mut self, max_frames: usize) {
        if max_frames == self.max_block_size {
            return;
        }
        let c = self.config.channel_width;
        self.max_block_size = max_frames;
        self.gather_buf = vec![0.0; max_frames * self.max_k_cin];
        self.buf_a      = vec![0.0; max_frames * c];
        self.buf_b      = vec![0.0; max_frames * c];
        self.res_buf    = vec![0.0; max_frames * c];
    }

    /// Number of conditioning parameters this model accepts.
    /// Wrappers must pass exactly this many floats to `update_conditioning`.
    pub fn nparams(&self) -> usize {
        self.config.nparams
    }

    /// Assert the model's `nparams` matches what a wrapper declares.
    ///
    /// Call after loading so a mismatched model drop-in (e.g. a 3-knob
    /// TubeScreamer JSON loaded by a 2-knob LA2A wrapper) fails loudly at init
    /// instead of silently producing garbage conditioning.
    pub fn require_nparams(&self, expected: usize) -> Result<(), String> {
        if self.config.nparams == expected {
            Ok(())
        } else {
            Err(format!(
                "loaded model has nparams={}, wrapper expects nparams={}",
                self.config.nparams, expected
            ))
        }
    }

    #[allow(dead_code)]
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn reset(&mut self) {
        for block in self.blocks.iter_mut() {
            block.ring.fill(0.0);
            block.prev_input.fill(0.0);
            block.ring_pos = 0;
        }
    }

    /// Refresh FiLM scale/shift for every block from the current parameter vector.
    ///
    /// `params.len()` must equal `self.nparams()`. Pass an empty slice for
    /// fixed-setting models (nparams == 0).
    pub fn update_conditioning(&mut self, params: &[f32]) {
        let n = self.config.nparams;
        debug_assert_eq!(
            params.len(), n,
            "update_conditioning: expected {} params, got {}",
            n, params.len(),
        );
        // gen MLP: nparams → 16 → 32 → 32 with ReLU between (and after the last).
        // For nparams = 0 the first layer's bias is the only signal, which matches
        // the training-time behavior where FiLM becomes a learned per-block bias.
        self.gen_scratch_a[..n].copy_from_slice(params);
        let mut in_len = n;
        let mut cur_in = &mut self.gen_scratch_a;
        let mut cur_out = &mut self.gen_scratch_b;
        for layer in self.gen.iter() {
            layer.forward(&cur_in[..in_len], &mut cur_out[..layer.out_features]);
            for v in cur_out[..layer.out_features].iter_mut() {
                if *v < 0.0 { *v = 0.0; }
            }
            in_len = layer.out_features;
            std::mem::swap(&mut cur_in, &mut cur_out);
        }
        debug_assert_eq!(in_len, 32);

        let c = self.config.channel_width;
        for block in self.blocks.iter_mut() {
            let n = block.bn_mean.len();
            block.adaptor.forward(&cur_in[..in_len], &mut self.adaptor_scratch[..2 * n]);
            for ch in 0..n {
                let g = self.adaptor_scratch[ch];
                let b = self.adaptor_scratch[ch + n];
                let s = g * block.bn_var_rsqrt[ch];
                block.film_scale[ch] = s;
                block.film_shift[ch] = b - block.bn_mean[ch] * s;
            }
            let _ = c;
        }
    }

    /// Process `samples` in-place. Block size must not exceed the value passed
    /// to `allocate_block_buffers` (default: 512).
    pub fn process_block_inplace(&mut self, samples: &mut [f32]) {
        let n = samples.len();
        debug_assert!(n <= self.max_block_size, "block size {} > max {}", n, self.max_block_size);

        let c = self.config.channel_width;

        // We ping-pong between buf_a and buf_b. `ping` is the index (0 or 1) of
        // the buffer that holds the current block's output after each pass.
        // Block 0 reads from `samples` directly (C_in = 1).
        // Blocks 1+ read from whichever buf was just written.
        let bufs: [*mut Vec<f32>; 2] = [
            &mut self.buf_a as *mut Vec<f32>,
            &mut self.buf_b as *mut Vec<f32>,
        ];
        let mut cur_out = 0usize; // index into bufs[] for the current output

        for (block_idx, block) in self.blocks.iter_mut().enumerate() {
            let k       = block.conv1_kernel;
            let in_ch   = block.conv1_in_channels;
            let out_ch  = block.conv1_out_channels;
            let ring_len = block.ring_len;
            let k_cin   = k * in_ch;

            let gather = &mut self.gather_buf[..n * k_cin];

            // ── Gather: write ring, fill gather[N, K×C_in] ──────────────────
            if block_idx == 0 {
                // C_in = 1: ring is [ring_len, 1] — scalar ring.
                for t in 0..n {
                    let rp = (block.ring_pos + t) % ring_len;
                    block.ring[rp] = samples[t];
                    let g_base = t * k;
                    for kk in 0..k {
                        let lag = block.conv1_lags[kk];
                        let idx = (rp + ring_len - lag) % ring_len;
                        gather[g_base + kk] = block.ring[idx];
                    }
                }
            } else {
                // C_in = channel_width (32): ring is [ring_len, C_in].
                // Reading tap k for a given position is a contiguous C_in-element slice.
                let prev_buf = unsafe { &*bufs[1 - cur_out] };
                for t in 0..n {
                    let rp = (block.ring_pos + t) % ring_len;
                    // Write: copy C_in channels at ring position rp.
                    let src = &prev_buf[t * in_ch .. (t + 1) * in_ch];
                    let dst = &mut block.ring[rp * in_ch .. (rp + 1) * in_ch];
                    dst.copy_from_slice(src);
                    // Gather K taps (each a C_in-element slice copy).
                    let g_base = t * k_cin;
                    for kk in 0..k {
                        let lag = block.conv1_lags[kk];
                        let idx = (rp + ring_len - lag) % ring_len;
                        let gsrc = &block.ring[idx * in_ch .. (idx + 1) * in_ch];
                        let gdst = &mut gather[g_base + kk * in_ch .. g_base + (kk + 1) * in_ch];
                        gdst.copy_from_slice(gsrc);
                    }
                }
            }
            block.ring_pos = (block.ring_pos + n) % ring_len;

            // ── SGEMM: gather[N, K×C_in] × W[K×C_in, C_out] → out[N, C_out] ─
            let out_buf = unsafe { &mut *bufs[cur_out] };
            let out_slice = &mut out_buf[..n * out_ch];
            sgemm(n, out_ch, k_cin, 1.0, gather, &block.conv1_weight, 0.0, out_slice);

            // ── Fused BN+FiLM + PReLU ────────────────────────────────────────
            // Must happen before the residual add — BN normalises the conv output
            // only; the residual bypass is added after activation (standard ResNet).
            {
                let fs = &block.film_scale;
                let fsh = &block.film_shift;
                let ps = &block.prelu_slope;
                for t in 0..n {
                    let row = &mut out_slice[t * out_ch .. (t + 1) * out_ch];
                    for ch in 0..out_ch {
                        let v = row[ch] * fs[ch] + fsh[ch];
                        row[ch] = if v < 0.0 { v * ps[ch] } else { v };
                    }
                }
            }

            // ── Residual ─────────────────────────────────────────────────────
            // Build res_buf[N, C_in]: shifted input (t-1), using prev_input for t=0.
            {
                let rb = &mut self.res_buf[..n * in_ch];
                rb[..in_ch].copy_from_slice(&block.prev_input[..in_ch]);
                if n > 1 {
                    if block_idx == 0 {
                        rb[in_ch..n].copy_from_slice(&samples[..n - 1]);
                    } else {
                        let prev_buf = unsafe { &*bufs[1 - cur_out] };
                        rb[in_ch..n * in_ch].copy_from_slice(&prev_buf[..(n - 1) * in_ch]);
                    }
                }
                // Save last input sample(s) for next buffer.
                if block_idx == 0 {
                    block.prev_input[0] = samples[n - 1];
                } else {
                    let prev_buf = unsafe { &*bufs[1 - cur_out] };
                    block.prev_input[..in_ch]
                        .copy_from_slice(&prev_buf[(n - 1) * in_ch .. n * in_ch]);
                }
            }
            if let Some(ref w) = block.res_dense {
                // [N, C_in] × [C_in, C_out] → add to out_slice (beta = 1.0)
                sgemm(n, out_ch, in_ch, 1.0, &self.res_buf[..n * in_ch], w, 1.0, out_slice);
            } else if let Some(ref w) = block.res_depthwise {
                // Element-wise: out[t, c] += res_buf[t, c] * w[c]
                let rb = &self.res_buf[..n * in_ch];
                for t in 0..n {
                    for ch in 0..out_ch {
                        out_slice[t * out_ch + ch] += rb[t * in_ch + ch] * w[ch];
                    }
                }
            }

            // Advance ping-pong. Block 0 wrote to buf[0]; next block reads buf[0].
            // For block_idx=0 we wrote to cur_out=0; next reads bufs[1-cur_out=1]?
            // Wait: we need next block to read what we just wrote (cur_out), so:
            // "prev_buf" for next iter = bufs[cur_out] (what we just wrote).
            // "out_buf"  for next iter = bufs[1 - cur_out].
            // So after writing to cur_out, flip so the *next* cur_out is the other buffer.
            cur_out = 1 - cur_out;
        }

        // ── Final output conv + tanh ──────────────────────────────────────────
        // The last block wrote to bufs[1 - cur_out] (we flipped after writing).
        let last_out = unsafe { &*bufs[1 - cur_out] };
        let ow = &self.output_weight;
        let ob = self.output_bias;
        for t in 0..n {
            let row = &last_out[t * c .. (t + 1) * c];
            let mut y = ob;
            for ch in 0..c {
                y += ow[ch] * row[ch];
            }
            samples[t] = y.tanh();
        }
    }

    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let mut buf = [input];
        self.process_block_inplace(&mut buf);
        buf[0]
    }

    pub fn receptive_field(&self) -> usize {
        self.config.receptive_field
    }
}

// ─── SGEMM wrapper ────────────────────────────────────────────────────────────
//
// C[M, N] = alpha * A[M, K] * B[K, N] + beta * C[M, N]   (all row-major)

#[cfg(target_os = "macos")]
#[inline]
fn sgemm(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32], beta: f32, c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::None,
            m as i32, n as i32, k as i32,
            alpha,
            a, k as i32,
            b, n as i32,
            beta,
            c, n as i32,
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[inline]
fn sgemm(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32], beta: f32, c: &mut [f32]) {
    if beta == 0.0 {
        c[..m * n].fill(0.0);
    } else if beta != 1.0 {
        for v in c[..m * n].iter_mut() { *v *= beta; }
    }
    for i in 0..m {
        for p in 0..k {
            let av = alpha * a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
}

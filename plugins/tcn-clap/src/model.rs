//! TCN inference runtime — pure Rust, allocation-free at audio rate.
//!
//! Loads JSON exported by `microtcn export` and runs the model sample-by-sample.
//! All weights are flattened into contiguous `Vec<f32>` on load so the hot path
//! is slice-linear and auto-vectorizable in `--release`.

use serde::Deserialize;
use std::path::Path;

// ---------- JSON schema (matches microtcn/export.py) --------------------------

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
    #[allow(dead_code)]
    nparams: usize,
    #[allow(dead_code)]
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

// ---------- Flat runtime representation --------------------------------------

struct Linear {
    weight: Vec<f32>, // row-major (out, in) → index: o * in_features + i
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
    // conv1: shape (out, in, kernel), laid out out-major → index (o * in + i) * K + k
    conv1_weight: Vec<f32>,
    conv1_in_channels: usize,
    conv1_kernel: usize,

    // Precomputed kernel taps: for each k, the offset-back in samples.
    // Used to turn the per-sample `(K-1-k)*d` calculation into a lookup.
    conv1_lags: Vec<usize>,

    bn_mean: Vec<f32>,
    bn_var_rsqrt: Vec<f32>,

    adaptor: Linear,

    prelu_slope: Vec<f32>,

    // Residual: either a dense 1×1 (groups=1, stored as flat (out, in)) or
    // a depthwise 1×1 (groups=in_channels, stored as per-channel scalars).
    res_dense: Option<Vec<f32>>,       // (out, in) flat, used when res_groups == 1
    res_depthwise: Option<Vec<f32>>,   // (out,) scalars, used otherwise
    res_in_channels: usize,

    // FiLM scale/shift refreshed on each param change.
    film_scale: Vec<f32>,
    film_shift: Vec<f32>,

    // Ring buffer for the dilated conv, shape (in_channels, ring_len), flat.
    ring: Vec<f32>,
    ring_len: usize,
    ring_pos: usize,

    // PyTorch's causal_crop drops the last sample of the residual, so block
    // output at time t uses res(input_{t-1}). We carry input_{t-1} here.
    prev_input: Vec<f32>,
}

pub struct TcnModel {
    config: ConfigJson,

    gen: Vec<Linear>,
    blocks: Vec<TcnBlock>,

    // Final 1×1 output conv: (1, channel_width) flat, plus optional scalar bias.
    output_weight: Vec<f32>,
    output_bias: f32,

    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
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

        let blocks = parsed
            .blocks
            .into_iter()
            .map(|b| {
                let k = b.conv1.kernel_size;
                let d = b.conv1.dilation;
                let in_ch = b.conv1.in_channels;
                let ring_len = (k - 1) * d + 1;

                // Store conv1 weights in (in, kernel, out) order — the inner
                // loop over o becomes a contiguous SIMD-friendly SAXPY.
                let out_ch = b.conv1.out_channels;
                let mut conv1_weight = vec![0.0_f32; out_ch * in_ch * k];
                for o in 0..out_ch {
                    for i in 0..in_ch {
                        for kk in 0..k {
                            let idx = (i * k + kk) * out_ch + o;
                            conv1_weight[idx] = b.conv1.weight[o][i][kk];
                        }
                    }
                }

                let conv1_lags: Vec<usize> = (0..k).map(|kk| (k - 1 - kk) * d).collect();

                let bn_var_rsqrt: Vec<f32> = b
                    .bn
                    .running_var
                    .iter()
                    .map(|v| 1.0 / (v + b.bn.eps).sqrt())
                    .collect();
                let num_features = b.bn.running_mean.len();

                let (res_dense, res_depthwise) = if b.res.groups == 1 {
                    let mut w = Vec::with_capacity(b.res.out_channels * b.res.in_channels);
                    for mat in &b.res.weight {
                        for row in mat {
                            w.push(row[0]);
                        }
                    }
                    (Some(w), None)
                } else {
                    // groups == in_channels == out_channels: weight shape (out, 1, 1),
                    // collapse to per-output-channel scalars.
                    let w: Vec<f32> = b.res.weight.iter().map(|m| m[0][0]).collect();
                    (None, Some(w))
                };

                TcnBlock {
                    conv1_weight,
                    conv1_in_channels: in_ch,
                    conv1_kernel: k,
                    conv1_lags,
                    bn_mean: b.bn.running_mean,
                    bn_var_rsqrt,
                    adaptor: Linear::from_json(b.adaptor),
                    prelu_slope: b.prelu.weight,
                    res_dense,
                    res_depthwise,
                    res_in_channels: b.res.in_channels,
                    film_scale: vec![1.0; num_features],
                    film_shift: vec![0.0; num_features],
                    ring: vec![0.0; in_ch * ring_len],
                    ring_len,
                    ring_pos: 0,
                    prev_input: vec![0.0; in_ch],
                }
            })
            .collect();

        // Output conv: (1, C, 1) → just a C-length weight vector + scalar bias.
        let output_weight: Vec<f32> = parsed.output.weight[0].iter().map(|row| row[0]).collect();
        let output_bias = parsed.output.bias.and_then(|v| v.first().copied()).unwrap_or(0.0);

        Ok(Self {
            config: parsed.config,
            gen,
            blocks,
            output_weight,
            output_bias,
            scratch_a: vec![0.0; channel_width],
            scratch_b: vec![0.0; channel_width],
            gen_scratch_a: vec![0.0; 32],
            gen_scratch_b: vec![0.0; 32],
            adaptor_scratch: vec![0.0; channel_width * 2],
        })
    }

    pub fn reset(&mut self) {
        for block in self.blocks.iter_mut() {
            block.ring.fill(0.0);
            block.prev_input.fill(0.0);
            block.ring_pos = 0;
        }
        self.scratch_a.fill(0.0);
        self.scratch_b.fill(0.0);
    }

    pub fn update_conditioning(&mut self, limit: f32, peak_reduction: f32) {
        // gen MLP: 2 → 16 → 32 → 32 with ReLU between (and after the last).
        self.gen_scratch_a[0] = limit;
        self.gen_scratch_a[1] = peak_reduction;
        let mut in_len = 2;
        let mut cur_in = &mut self.gen_scratch_a;
        let mut cur_out = &mut self.gen_scratch_b;
        for layer in self.gen.iter() {
            layer.forward(&cur_in[..in_len], &mut cur_out[..layer.out_features]);
            for v in cur_out[..layer.out_features].iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            in_len = layer.out_features;
            std::mem::swap(&mut cur_in, &mut cur_out);
        }
        debug_assert_eq!(in_len, 32);

        for block in self.blocks.iter_mut() {
            let n = block.bn_mean.len();
            block.adaptor.forward(
                &cur_in[..in_len],
                &mut self.adaptor_scratch[..2 * n],
            );
            for c in 0..n {
                let g = self.adaptor_scratch[c];
                let b = self.adaptor_scratch[c + n];
                let s = g * block.bn_var_rsqrt[c];
                block.film_scale[c] = s;
                block.film_shift[c] = b - block.bn_mean[c] * s;
            }
        }
    }

    #[inline]
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let channel_width = self.config.channel_width;
        let Self {
            blocks,
            scratch_a,
            scratch_b,
            output_weight,
            output_bias,
            ..
        } = self;

        scratch_a[0] = input;
        let mut x_len: usize = 1;

        for block in blocks.iter_mut() {
            let k = block.conv1_kernel;
            let in_ch = block.conv1_in_channels;
            let ring_len = block.ring_len;
            let ring_pos = block.ring_pos;

            // Push current input into the ring (one slot per input channel).
            for c in 0..x_len {
                block.ring[c * ring_len + ring_pos] = scratch_a[c];
            }

            // Precompute the K ring indices for this sample. Done once instead
            // of K times inside the (o, i) loop.
            let mut idx_buf = [0usize; 32]; // K ≤ 32 in practice
            for kk in 0..k {
                let lag = block.conv1_lags[kk];
                idx_buf[kk] = (ring_pos + ring_len - lag) % ring_len;
            }

            // Dilated conv. Weights are laid out (in, kernel, out); loops are
            // ordered so the innermost `o` axis hits contiguous weight bytes
            // and a contiguous scratch write — easy to auto-vectorize and
            // amortizes each ring read across all 32 output channels.
            for o in 0..channel_width {
                scratch_b[o] = 0.0;
            }
            for i in 0..in_ch {
                let ring_row = &block.ring[i * ring_len..(i + 1) * ring_len];
                for kk in 0..k {
                    let x = ring_row[idx_buf[kk]];
                    let w_base = (i * k + kk) * channel_width;
                    let w_slice = &block.conv1_weight[w_base..w_base + channel_width];
                    for o in 0..channel_width {
                        scratch_b[o] += w_slice[o] * x;
                    }
                }
            }

            block.ring_pos = (ring_pos + 1) % ring_len;

            // Fused BN + FiLM affine.
            for c in 0..channel_width {
                scratch_b[c] = scratch_b[c] * block.film_scale[c] + block.film_shift[c];
            }

            // PReLU.
            for c in 0..channel_width {
                if scratch_b[c] < 0.0 {
                    scratch_b[c] *= block.prelu_slope[c];
                }
            }

            // Residual from previous sample's input (matches PyTorch causal_crop).
            if let Some(ref w) = block.res_dense {
                let in_n = block.res_in_channels;
                for o in 0..channel_width {
                    let w_row = &w[o * in_n..(o + 1) * in_n];
                    let mut acc = 0.0_f32;
                    for i in 0..in_n {
                        acc += w_row[i] * block.prev_input[i];
                    }
                    scratch_b[o] += acc;
                }
            } else if let Some(ref w) = block.res_depthwise {
                for o in 0..channel_width {
                    scratch_b[o] += w[o] * block.prev_input[o];
                }
            }

            // Update prev_input = current block input (before the swap).
            block.prev_input[..x_len].copy_from_slice(&scratch_a[..x_len]);

            std::mem::swap(scratch_a, scratch_b);
            x_len = channel_width;
        }

        // Final 1×1 output → tanh.
        let mut y = *output_bias;
        for i in 0..x_len {
            y += output_weight[i] * scratch_a[i];
        }
        y.tanh()
    }

    #[allow(dead_code)]
    pub fn receptive_field(&self) -> usize {
        self.config.receptive_field
    }
}

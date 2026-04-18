//! TCN inference runtime (pure Rust, allocation-free at audio rate).
//!
//! Loads a JSON exported by `microtcn export` and runs the model sample-by-sample.
//! The JSON holds weights for: a 3-layer conditioning MLP, N TCN blocks (each
//! with a dilated conv, BN stats, adaptor MLP for FiLM, PReLU, and a residual
//! 1×1 conv), and a final 1×1 output conv.
//!
//! All per-sample work is bounds-checked slice indexing — the compiler
//! hoists the checks for tight inner loops in `--release`. No `unsafe`.

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

// ---------- Runtime representation -------------------------------------------

struct TcnBlock {
    conv1_weight: Vec<Vec<Vec<f32>>>, // (out, in, kernel)
    conv1_in_channels: usize,
    conv1_kernel: usize,
    conv1_dilation: usize,

    bn_mean: Vec<f32>,
    bn_var_rsqrt: Vec<f32>, // precomputed 1 / sqrt(var + eps)

    adaptor_weight: Vec<Vec<f32>>, // (out=2*num_features, in=32)
    adaptor_bias: Vec<f32>,

    prelu_slope: Vec<f32>, // per-channel

    res_weight: Vec<Vec<Vec<f32>>>, // (out, in/groups, kernel=1)
    res_groups: usize,

    // FiLM scale/shift, refreshed on each param change.
    film_scale: Vec<f32>, // g[c] / sqrt(var[c] + eps)
    film_shift: Vec<f32>, // b[c] - mean[c] * film_scale[c]

    // Ring buffer for the dilated conv — (kernel-1)*dilation+1 samples per input channel.
    ring: Vec<Vec<f32>>, // [in_channels][ring_len]
    ring_len: usize,
    ring_pos: usize,

    // Input from the PREVIOUS sample, used for the residual add. This mirrors
    // PyTorch's `causal_crop(x_res, L)` which drops x_res[-1], so the block's
    // conv output at time t is added to res(input_{t-1}). Without this, streaming
    // output drifts from PyTorch by ~1 sample's worth of residual per block.
    prev_input: Vec<f32>,
}

pub struct TcnModel {
    config: ConfigJson,

    // Conditioning MLP: three Linears (Linear → ReLU → Linear → ReLU → Linear → ReLU).
    gen_layers: Vec<LinearJson>,

    blocks: Vec<TcnBlock>,

    output_weight: Vec<Vec<Vec<f32>>>, // (1, channel_width, 1)
    output_bias: Option<Vec<f32>>,

    // Per-sample scratch — sized to max channel width; allocated once on load.
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
    gen_scratch_in: Vec<f32>,
    gen_scratch_out: Vec<f32>,
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

        let blocks = parsed
            .blocks
            .into_iter()
            .map(|b| {
                let ring_len = (b.conv1.kernel_size - 1) * b.conv1.dilation + 1;
                let bn_var_rsqrt: Vec<f32> = b
                    .bn
                    .running_var
                    .iter()
                    .map(|v| 1.0 / (v + b.bn.eps).sqrt())
                    .collect();
                let num_features = b.bn.running_mean.len();
                TcnBlock {
                    conv1_weight: b.conv1.weight,
                    conv1_in_channels: b.conv1.in_channels,
                    conv1_kernel: b.conv1.kernel_size,
                    conv1_dilation: b.conv1.dilation,
                    bn_mean: b.bn.running_mean,
                    bn_var_rsqrt,
                    adaptor_weight: b.adaptor.weight,
                    adaptor_bias: b.adaptor.bias.unwrap_or_else(|| vec![0.0; b.adaptor.out_features]),
                    prelu_slope: b.prelu.weight,
                    res_weight: b.res.weight,
                    res_groups: b.res.groups,
                    film_scale: vec![1.0; num_features],
                    film_shift: vec![0.0; num_features],
                    ring: vec![vec![0.0; ring_len]; b.conv1.in_channels],
                    ring_len,
                    ring_pos: 0,
                    prev_input: vec![0.0; b.conv1.in_channels],
                }
            })
            .collect();

        let largest_adaptor_out = parsed.config.channel_width * 2; // adaptor outputs (g, b), each C

        Ok(Self {
            config: parsed.config,
            gen_layers: parsed.gen,
            blocks,
            output_weight: parsed.output.weight,
            output_bias: parsed.output.bias,
            scratch_a: vec![0.0; channel_width],
            scratch_b: vec![0.0; channel_width],
            gen_scratch_in: vec![0.0; 32], // gen MLP hits at most 32-d intermediates
            gen_scratch_out: vec![0.0; 32],
            adaptor_scratch: vec![0.0; largest_adaptor_out],
        })
    }

    pub fn reset(&mut self) {
        for block in self.blocks.iter_mut() {
            for ch in block.ring.iter_mut() {
                ch.fill(0.0);
            }
            block.ring_pos = 0;
            block.prev_input.fill(0.0);
        }
        self.scratch_a.fill(0.0);
        self.scratch_b.fill(0.0);
    }

    /// Refresh FiLM conditioning from the two compressor params. Cheap — runs
    /// the 3-layer gen MLP once, then the adaptor per block, and folds BN +
    /// affine into per-channel scale / shift.
    pub fn update_conditioning(&mut self, limit: f32, peak_reduction: f32) {
        // gen MLP: 2 → 16 → 32 → 32, ReLU between layers.
        let mut cur_in = &mut self.gen_scratch_in;
        let mut cur_out = &mut self.gen_scratch_out;
        cur_in[0] = limit;
        cur_in[1] = peak_reduction;
        let mut in_len = 2;

        for layer in self.gen_layers.iter() {
            linear_forward_into(
                &cur_in[..in_len],
                &layer.weight,
                layer.bias.as_deref(),
                &mut cur_out[..layer.out_features],
            );
            // ReLU
            for v in cur_out[..layer.out_features].iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            in_len = layer.out_features;
            std::mem::swap(&mut cur_in, &mut cur_out);
        }
        // After the loop, cur_in holds the 32-d conditioning vector.
        debug_assert_eq!(in_len, 32);

        // Adaptor per block: (32) → (2 * num_features). Fuse into film_scale / film_shift.
        for block in self.blocks.iter_mut() {
            let n = block.bn_mean.len();
            linear_forward_into(
                &cur_in[..in_len],
                &block.adaptor_weight,
                Some(&block.adaptor_bias),
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

    /// Per-sample TCN forward pass. No heap allocations; ring buffers carry state.
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

        // Seed block-0 input.
        scratch_a[0] = input;
        let mut x_len: usize = 1;

        for block in blocks.iter_mut() {
            // Push current input into the block's ring buffer.
            for c in 0..x_len {
                block.ring[c][block.ring_pos] = scratch_a[c];
            }

            // Dilated conv: produce channel_width output channels.
            // weight[o][i][k]: k=0 is the OLDEST kernel tap, k=K-1 is the NEWEST.
            for o in 0..channel_width {
                let w_o = &block.conv1_weight[o];
                let mut acc = 0.0_f32;
                for i in 0..block.conv1_in_channels {
                    let w_oi = &w_o[i];
                    let ring_i = &block.ring[i];
                    for k in 0..block.conv1_kernel {
                        let lag = (block.conv1_kernel - 1 - k) * block.conv1_dilation;
                        let idx = (block.ring_pos + block.ring_len - lag) % block.ring_len;
                        acc += w_oi[k] * ring_i[idx];
                    }
                }
                scratch_b[o] = acc;
            }

            // Advance the write head.
            block.ring_pos = (block.ring_pos + 1) % block.ring_len;

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

            // Residual 1×1 conv applied to the PREVIOUS sample's input
            // (block.prev_input), matching PyTorch's causal_crop off-by-one.
            // groups=1: regular 1×1, (out=C, in=in_channels, kernel=1)
            // groups=in_channels: depthwise, one weight per output channel.
            if block.res_groups == 1 {
                for o in 0..channel_width {
                    let w_o = &block.res_weight[o];
                    let mut acc = 0.0_f32;
                    for i in 0..block.prev_input.len() {
                        acc += w_o[i][0] * block.prev_input[i];
                    }
                    scratch_b[o] += acc;
                }
            } else {
                for o in 0..channel_width {
                    scratch_b[o] += block.res_weight[o][0][0] * block.prev_input[o];
                }
            }

            // Save the current block input as prev_input for the next call.
            // `scratch_a[0..x_len]` is still the block's input at this point.
            for c in 0..x_len {
                block.prev_input[c] = scratch_a[c];
            }

            // scratch_a becomes the block's output for the next iteration.
            std::mem::swap(scratch_a, scratch_b);
            x_len = channel_width;
        }

        // Final 1×1 output conv → tanh.
        let mut y = 0.0_f32;
        let w_out = &output_weight[0]; // (in, 1)
        for i in 0..x_len {
            y += w_out[i][0] * scratch_a[i];
        }
        if let Some(bias) = output_bias.as_deref() {
            y += bias[0];
        }
        y.tanh()
    }

    #[allow(dead_code)]
    pub fn receptive_field(&self) -> usize {
        self.config.receptive_field
    }
}

// ---------- Helpers ----------------------------------------------------------

fn linear_forward_into(x: &[f32], weight: &[Vec<f32>], bias: Option<&[f32]>, out: &mut [f32]) {
    // weight: (out_features, in_features). out[o] = sum_i w[o][i] * x[i] (+ bias[o])
    debug_assert_eq!(out.len(), weight.len());
    for o in 0..out.len() {
        let w_row = &weight[o];
        let mut acc = 0.0_f32;
        for i in 0..x.len() {
            acc += w_row[i] * x[i];
        }
        if let Some(b) = bias {
            acc += b[o];
        }
        out[o] = acc;
    }
}

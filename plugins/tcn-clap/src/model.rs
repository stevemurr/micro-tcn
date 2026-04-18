//! TCN inference runtime (pure Rust, allocation-free at audio rate).
//!
//! Loads a JSON exported by `microtcn export` and runs the model sample-by-sample.
//! The JSON holds weights for: a 3-layer conditioning MLP, N TCN blocks
//! (each with a dilated conv, a fused BN, an adaptor MLP, PReLU, and a residual
//! 1×1 conv), and a final 1×1 output conv.
//!
//! Current status: parses the JSON, carves out ring buffers, implements the
//! conditioning update path, and leaves `process_sample` as a passthrough. All
//! the weights and static sizes are available — filling in the conv / BN / PReLU
//! arithmetic is the remaining work.

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

#[derive(Deserialize)]
struct ConfigJson {
    nblocks: usize,
    kernel_size: usize,
    dilation_growth: usize,
    channel_width: usize,
    causal: bool,
    #[allow(dead_code)]
    nparams: usize,
    #[allow(dead_code)]
    sample_rate: u32,
    receptive_field: usize,
}

#[derive(Deserialize)]
struct LinearJson {
    in_features: usize,
    out_features: usize,
    weight: Vec<Vec<f32>>, // (out, in)
    bias: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct Conv1dJson {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
    weight: Vec<Vec<Vec<f32>>>, // (out, in/groups, kernel)
    bias: Option<Vec<f32>>,
}

#[derive(Deserialize)]
struct Bn1dJson {
    num_features: usize,
    #[allow(dead_code)]
    affine: bool,
    eps: f32,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
}

#[derive(Deserialize)]
struct PreluJson {
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
    conv1_dilation: usize,
    conv1_kernel: usize,

    bn_mean: Vec<f32>,
    bn_var_rsqrt: Vec<f32>, // pre-computed 1/sqrt(var + eps)

    adaptor_weight: Vec<Vec<f32>>, // (out, in)
    adaptor_bias: Vec<f32>,

    prelu_slope: Vec<f32>,

    res_weight: Vec<Vec<Vec<f32>>>, // (out, in/groups, kernel=1)
    res_groups: usize,

    // FiLM scale/shift, refreshed on each param change.
    film_scale: Vec<f32>, // per-channel (g[c] / sqrt(var[c] + eps))
    film_shift: Vec<f32>, // per-channel (b[c] - mean[c] * film_scale[c])

    // Ring buffer for the dilated conv (sized for (kernel - 1) * dilation + 1 samples
    // per input channel).
    ring: Vec<Vec<f32>>, // [in_channels][ring_len]
    ring_len: usize,
    ring_pos: usize,
}

pub struct TcnModel {
    config: ConfigJson,

    // Conditioning MLP: three Linears (Linear → ReLU → Linear → ReLU → Linear → ReLU).
    gen_layers: Vec<LinearJson>,

    blocks: Vec<TcnBlock>,

    output_weight: Vec<Vec<Vec<f32>>>, // (out, in, kernel=1)
    output_bias: Option<Vec<f32>>,
}

impl TcnModel {
    pub fn load_from_json(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let parsed: ModelJson = serde_json::from_str(&text).map_err(|e| e.to_string())?;

        if parsed.version != 1 {
            return Err(format!("unsupported export version {}", parsed.version));
        }
        if parsed.arch != "direct" {
            return Err(format!("only arch='direct' is supported, got {}", parsed.arch));
        }

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
                let n_channels = b.bn.num_features;
                TcnBlock {
                    conv1_weight: b.conv1.weight,
                    conv1_dilation: b.conv1.dilation,
                    conv1_kernel: b.conv1.kernel_size,
                    bn_mean: b.bn.running_mean,
                    bn_var_rsqrt,
                    adaptor_weight: b.adaptor.weight,
                    adaptor_bias: b.adaptor.bias.unwrap_or_else(|| vec![0.0; b.adaptor.out_features]),
                    prelu_slope: b.prelu.weight,
                    res_weight: b.res.weight,
                    res_groups: b.res.groups,
                    film_scale: vec![1.0; n_channels],
                    film_shift: vec![0.0; n_channels],
                    ring: vec![vec![0.0; ring_len]; b.conv1.in_channels],
                    ring_len,
                    ring_pos: 0,
                }
            })
            .collect();

        Ok(Self {
            config: parsed.config,
            gen_layers: parsed.gen,
            blocks,
            output_weight: parsed.output.weight,
            output_bias: parsed.output.bias,
        })
    }

    /// Clears every ring buffer. Call on `Plugin::reset`.
    pub fn reset(&mut self) {
        for block in self.blocks.iter_mut() {
            for ch in block.ring.iter_mut() {
                ch.fill(0.0);
            }
            block.ring_pos = 0;
        }
    }

    /// Refresh FiLM conditioning from the two compressor params. Cheap —
    /// runs the 3-layer gen MLP once, then the adaptor in each block, and
    /// fuses BN + affine into per-channel scale / shift.
    pub fn update_conditioning(&mut self, limit: f32, peak_reduction: f32) {
        // Run gen MLP on [limit, peak_reduction]. Expected shape: (nparams=2) → 32.
        let mut cond: Vec<f32> = vec![limit, peak_reduction];
        for layer in self.gen_layers.iter() {
            cond = linear_relu(&cond, &layer.weight, layer.bias.as_deref());
        }
        debug_assert_eq!(cond.len(), 32, "gen MLP should produce a 32-d conditioning vector");

        // For each block: adaptor(cond) → (g, b) each of length num_features;
        // fuse with BN running stats into film_scale / film_shift.
        for block in self.blocks.iter_mut() {
            let gb = linear_forward(&cond, &block.adaptor_weight, Some(&block.adaptor_bias));
            let n = block.bn_mean.len();
            debug_assert_eq!(gb.len(), 2 * n, "adaptor should output 2 × num_features");
            for c in 0..n {
                let g = gb[c];
                let b = gb[c + n];
                let s = g * block.bn_var_rsqrt[c];
                block.film_scale[c] = s;
                block.film_shift[c] = b - block.bn_mean[c] * s;
            }
        }
    }

    /// Per-sample forward pass. Currently passthrough — implement the TCN
    /// forward here sample-by-sample using the ring buffers already allocated.
    ///
    /// Sketch:
    ///   x = [sample]                     // shape (1,)
    ///   for block in blocks:
    ///       x_in = x (before ring push)
    ///       push x into block.ring (per input channel, at block.ring_pos)
    ///       conv_out[o] = sum over i, k of conv1_weight[o, i, k] *
    ///                     block.ring[i][(ring_pos - k * dilation) mod ring_len]
    ///       x = conv_out * film_scale + film_shift       // fused BN + FiLM affine
    ///       x = prelu(x, prelu_slope)
    ///       res_out = apply res conv (1×1, grouped=in_channels) to x_in
    ///       x = x + res_out                               // residual add
    ///       advance block.ring_pos by 1 (mod ring_len)
    ///   output = linear-project x (32 → 1), add bias
    ///   return tanh(output)
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // TODO: implement the TCN forward pass. For now, passthrough so the
        // plugin loads and the GUI / param plumbing is verifiable end-to-end.
        let _ = (&self.config, &self.blocks, &self.output_weight, &self.output_bias);
        input
    }

    pub fn receptive_field(&self) -> usize {
        self.config.receptive_field
    }
}

// ---------- Small linear-algebra primitives (no allocation in the hot path) ---

fn linear_forward(x: &[f32], weight: &[Vec<f32>], bias: Option<&[f32]>) -> Vec<f32> {
    // weight: (out, in). Returns (out,).
    let out_features = weight.len();
    let mut y = vec![0.0_f32; out_features];
    for o in 0..out_features {
        let w_row = &weight[o];
        let mut acc = 0.0_f32;
        for i in 0..x.len() {
            acc += w_row[i] * x[i];
        }
        if let Some(b) = bias {
            acc += b[o];
        }
        y[o] = acc;
    }
    y
}

fn linear_relu(x: &[f32], weight: &[Vec<f32>], bias: Option<&[f32]>) -> Vec<f32> {
    let mut y = linear_forward(x, weight, bias);
    for v in y.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    y
}

#[allow(dead_code)]
fn prelu(x: &mut [f32], slope: &[f32]) {
    for (v, s) in x.iter_mut().zip(slope.iter()) {
        if *v < 0.0 {
            *v *= *s;
        }
    }
}

#[allow(dead_code)]
fn tanh_scalar(x: f32) -> f32 {
    x.tanh()
}

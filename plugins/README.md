# micro-TCN plugins

## Export a checkpoint

From the repo root:

```
uv run microtcn export \
  --checkpoint /path/to/checkpoints/step=NNN-val=X.YYY.ckpt \
  --output /path/to/tcn.json
```

Only `--arch direct` checkpoints are currently exportable. The output is a plain
JSON with all conv weights, BN stats, PReLU slopes, adaptor MLP, and the final
conv — everything needed for inference.

## tcn-clap (nih-plug, Rust)

A CLAP + VST3 plugin that loads an exported `tcn.json` and runs the TCN in
real-time. Built on [nih-plug](https://github.com/robbert-vdh/nih-plug).

### Architecture

The plugin uses a **pure Rust** inference runtime (no RTNeural / libtorch / ONNX
dependency). Why:

- Real-time safe by construction — no heap allocations during `process()`
- Small binary (~3 MB vs. 500 MB for libtorch)
- Easier cross-platform builds (no C++ FFI)
- The model is small enough (50 K params, 4 blocks) that hand-rolled Rust
  matches RTNeural's perf without the template gymnastics

The inference module (`src/model.rs`) loads the JSON once on init, lays out
ring buffers for dilated convs, and processes audio sample-by-sample in the
audio callback. Parameters (`peak_reduction`, `limit`) trigger a cheap update
of the FiLM conditioning (re-runs the 3-layer MLP → caches scale/shift per
block) on change; the per-sample hot path has no MLP forward.

### Build

```
cd plugins/tcn-clap
cargo xtask bundle tcn_clap --release
```

Output lands in `target/bundled/`. The CLAP file is `tcn_clap.clap`;
copy it to your CLAP path (e.g. `~/.clap/` on Linux, `~/Library/Audio/Plug-Ins/CLAP/` on macOS).

### Shipping a specific model

At runtime the plugin looks for `tcn.json` next to the plugin binary (or via the
`TCN_CLAP_MODEL` env var). Replace that file to swap the model without rebuilding.

### Status

- [x] nih-plug scaffold, CLAP + VST3 output
- [x] Parameters: `peak_reduction` (0..1), `limit` (0 = compress / 1 = limit)
- [x] JSON model loader
- [ ] Dilated Conv1D forward pass (stub — currently passthrough)
- [ ] FiLM conditioning update on param change (stub)
- [ ] Output tanh + residuals
- [ ] Latency reporting / warm-up handling

The DSP stub is in `src/model.rs` with comments marking exactly where each
layer's arithmetic goes. The JSON schema is already parsed, so all weights are
available — this is purely a matter of filling in the conv / BN / PReLU math.

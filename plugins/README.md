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

Output lands in `target/bundled/` — `tcn_clap.clap` and `tcn_clap.vst3`.

### macOS — one-command build + install

```
cd plugins/tcn-clap
./install-macos.sh                     # uses the baked-in default model
./install-macos.sh /path/to/tcn.json   # override with a different checkpoint
```

Builds via `cargo xtask bundle`, copies into `~/Library/Audio/Plug-Ins/{CLAP,VST3}/`,
and re-codesigns (ad-hoc) — the codesign step is required after any bundle content
change or macOS will silently refuse to load the plugin.

### Model resolution (at plugin init)

In priority order:

1. `TCN_CLAP_MODEL` env var — explicit path override
2. macOS: `Contents/Resources/tcn.json` inside the bundle (standard Apple convention, sealed by codesign)
3. Linux/Windows: `tcn.json` next to the plugin file
4. Baked-in default at `plugins/tcn-clap/assets/tcn.json`, embedded at compile time via `include_str!()`

The baked-in default means a freshly built plugin works out of the box with
no separate file to copy. Replace `assets/tcn.json` and rebuild to change the
default, or use the env var / Resources override to ship alternate models.

### Weight checkin

`assets/tcn.json` is checked into the repo (~1.1 MB JSON). This keeps the
plugin self-contained and means any build produces a working binary. If the
file grows (bigger models) or we accumulate many checkpoints, moving to git-lfs
or a binary export format would be the next step.

### Status

- [x] nih-plug scaffold, CLAP + VST3 output
- [x] Parameters: `peak_reduction` (0..1), `limit` (0 = compress / 1 = limit)
- [x] JSON model loader
- [x] Dilated Conv1D forward pass (ring-buffer sample-by-sample)
- [x] FiLM conditioning update on param change (fused BN + affine, recomputed on param change)
- [x] Residual 1×1 conv (groups=1 for block 0, depthwise for blocks 1..N)
- [x] PReLU activation, final tanh head
- [ ] Numerical validation against PyTorch reference
- [ ] Warm-up latency handling — first `receptive_field` samples will have cold-ring-buffer artifacts

### Validating the port

Cheapest test: after building and installing the plugin, load it on a signal
generator track in a DAW at 44.1 kHz, compare its output against the
`microtcn comp` CLI output for the same input file and param settings. They
should be sample-accurate modulo float ordering.

If they differ, the likely suspects are (1) the `causal_crop` off-by-one in the
original PyTorch reference — the plugin currently uses "aligned" residual
(same-sample add), while PyTorch's crop adds the *previous* sample's residual.
Impact is ~22 µs mis-alignment per block, probably inaudible, but to match
training exactly you'd add a 1-sample delay to each block's residual.

# micro-TCN plugins

## Workspace layout

```
plugins/
├── Cargo.toml               # workspace root
├── install-macos.sh         # build + install every plugin (or a subset)
├── tcn-plugin-core/         # shared: TcnModel runtime + model-path resolver
├── tcn-la2a/                # wrapper — LA2A compressor (2 knobs)
├── tcn-tubescreamer/        # wrapper — tube-screamer distortion (0 knobs)
└── xtask/                   # `cargo xtask bundle <crate>`
```

The inference runtime (pure Rust, allocation-free at audio rate) lives in
`tcn-plugin-core`. Each per-model wrapper only contributes a
`#[derive(Params)]` struct, its CLAP/VST3 identity, and the mapping from its
declared knobs to the flat `&[f32]` the model's FiLM conditioning expects.

## Export a checkpoint

From the repo root:

```
uv run microtcn export \
  --checkpoint /path/to/checkpoints/step=NNN-val=X.YYY.ckpt \
  --output /path/to/tcn.json
```

Only `--arch direct` checkpoints are currently exportable. The JSON includes
`config.nparams`, which wrappers validate against at load time — a 3-knob
model dropped into the 2-knob LA2A wrapper fails loudly instead of silently
producing garbage conditioning.

## Building

From the `plugins/` workspace root:

```
cargo xtask bundle tcn-la2a --release
```

Output lands in `plugins/target/bundled/` (workspace target, not per-crate).

## Install on macOS

One unified installer at the workspace root handles every plugin:

```
cd plugins
./install-macos.sh                              # install every plugin
./install-macos.sh tcn-la2a                     # install one
./install-macos.sh tcn-la2a tcn-tubescreamer    # install a subset

# override the bundled model per-plugin via env vars (absent = bundled default):
TCN_LA2A_MODEL=/path/to/la2a.json \
TCN_TUBESCREAMER_MODEL=/path/to/ts.json \
  ./install-macos.sh
```

The script builds via `cargo xtask bundle <crate>`, copies into
`~/Library/Audio/Plug-Ins/{CLAP,VST3}/`, and re-codesigns (ad-hoc) — the
codesign step is required after any bundle-content change or macOS silently
refuses to load the plugin.

## Model resolution (at plugin init)

Priority order, consulted by `tcn_plugin_core::locate_model()`:

1. **Per-plugin env var override** — e.g. `TCN_LA2A_MODEL=/path/to/tcn.json`.
   Each wrapper declares its own variable so two plugins installed side by
   side don't collide.
2. **macOS bundle**: `Contents/Resources/tcn.json` inside the `.clap`/`.vst3`
   (standard Apple convention, sealed by codesign).
3. **Linux/Windows**: `tcn.json` next to the plugin binary.
4. **Embedded default** baked in at compile time from
   `plugins/<wrapper>/assets/tcn.json`.

## Adding a new per-model wrapper

Concrete procedure (rough template — the LA2A crate is ~170 lines total):

1. Export your trained checkpoint to `plugins/tcn-<name>/assets/tcn.json`.
2. Copy `plugins/tcn-la2a/` to `plugins/tcn-<name>/`; `git mv` keeps history.
3. In `plugins/tcn-<name>/Cargo.toml`: rename the `[package]` name.
4. In `plugins/tcn-<name>/src/lib.rs`:
   - Rename the plugin struct, `NPARAMS`, and the `MODEL_ENV_OVERRIDE`
     constant (e.g. `TCN_TUBESCREAMER_MODEL`).
   - Declare one `FloatParam`/`BoolParam` per knob the model was trained with.
     Match names and ranges to the dataset's captured knob axes — the
     `param_names` field in the dataset metadata is the source of truth.
   - Update the `update_conditioning(&self.cond_scratch)` call: `cond_scratch`
     must be `[f32; NPARAMS]`, filled in the order the model was trained with.
   - Update CLAP_ID, CLAP_DESCRIPTION, CLAP_FEATURES (e.g.
     `ClapFeature::Distortion` for a tube screamer), and VST3_CLASS_ID (any
     unique 16-byte literal).
5. Add the crate to `plugins/Cargo.toml`'s `members`, and add the crate name
   + env-var-to-plugin mapping to `install-macos.sh` (`ALL_PLUGINS` and the
   `env_var_for` case).
6. `cargo xtask bundle tcn-<name> --release` to sanity-check.

If the wrapper's declared `NPARAMS` doesn't match the loaded model's
`config.nparams`, `model.require_nparams(NPARAMS)` logs a clear error in
`initialize()` and the plugin refuses to load — catches a lot of the
"dropped the wrong model JSON into Resources/" failures.

### Special case: nparams = 0

A model trained on a fixed-setting capture (e.g. the current TubeScreamer
dataset, recorded at a single drive position) has `nparams = 0`. The wrapper
declares zero knobs, `cond_scratch` is `[f32; 0]`, and `update_conditioning`
is called with an empty slice. FiLM effectively becomes a learned per-block
bias — exactly the training-time behavior.

## Status (LA2A)

- [x] nih-plug scaffold, CLAP + VST3 output
- [x] Parameters: `peak_reduction` (0..1), `limit` (0 = compress / 1 = limit)
- [x] JSON model loader, dilated Conv1D forward, FiLM conditioning
- [x] Residual 1×1 conv (groups=1 for block 0, depthwise for blocks 1..N)
- [x] PReLU activation, final tanh head
- [x] `nparams` validation at init
- [ ] Numerical validation against PyTorch reference
- [ ] Warm-up latency handling — first `receptive_field` samples have
      cold-ring-buffer artifacts

## Weight checkin

Each wrapper's `assets/tcn.json` is checked into the repo (~1.1 MB JSON
each). Keeps plugin crates self-contained and means any build produces a
working binary. If this grows painful (many wrappers, bigger models), moving
to git-lfs or a binary export format is the next step.

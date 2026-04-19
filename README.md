<div align="center">

# micro-TCN
| [Paper](https://arxiv.org/abs/2102.06200) | [Demo](https://csteinmetz1.github.io/tcn-audio-effects/) |

Efficient neural networks for real-time modeling of analog dynamic range compression.

</div>

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):
```
uv sync
```

## Layout

```
microtcn/
  data.py      SignalTrain LA2A dataset (int16 mmap cache → [-1, 1] float)
  model.py     TCN with FiLM conditioning (tanh output)
  loss.py      L1 + STFT magnitude
  train.py     training loop
  eval.py      checkpoint evaluation, per-param-class metrics
  comp.py      run a trained checkpoint over a WAV file
  cli.py       typer entry points
  utils.py     causal_crop, center_crop
```

Checkpoints are plain `torch.save` dicts: `{"model": state_dict, "config": {...}, "epoch": N, "val_loss": f}`.

## Training

```
uv run microtcn train \
  --root-dir /path/to/SignalTrain_LA2A_Dataset_1.1 \
  --artifact-dir ./runs/uTCN-300 \
  --nblocks 4 --dilation-growth 10 --kernel-size 13 --causal \
  --arch hybrid \
  --batch-size 16 --lr 1e-3 \
  --max-steps 20000 --warmup-steps 500 --eval-every 1000 \
  --precision bf16 --num-workers 4
```

First run decodes every WAV into `<root-dir>/.cache/{subset}_{input,target}.bin` (one-time, ~14 GB). Later runs mmap it directly. Training is step-based: linear warmup over `--warmup-steps`, cosine decay to 0 by `--max-steps`. Validation runs every `--eval-every` steps; pass `--val-max-batches N` to cap val batches per eval for faster feedback. Per-eval train/val losses plus `α` (coloration gate for the hybrid arch) are written to `<artifact-dir>/log.csv`; top-k and `last.ckpt` land in `<artifact-dir>/checkpoints/`.

**Architectures** (`--arch`):
- `direct` — `y = tanh(conv(features))`. Free-form prediction, original paper's head.
- `hybrid` — `y = sigmoid(g) · x + α · tanh(d)`. Gain modulator + small learned coloration residual; `α` starts at 0 and grows only if the loss demands signal-path corrections the gain branch can't explain. Embeds compressor physics as a structural prior, converges faster and more reliably with limited data.

`--precision` accepts `fp32`, `bf16`, or `fp16` (controls `torch.autocast`).

## Evaluation

```
uv run microtcn eval \
  --root-dir /path/to/SignalTrain_LA2A_Dataset_1.1 \
  --checkpoint ./runs/uTCN-300/checkpoints/last.ckpt \
  --subset val
```

Prints per-param-class L1, STFT, and aggregate losses.

## Processing a file

```
uv run microtcn comp \
  --checkpoint ./runs/uTCN-300/checkpoints/last.ckpt \
  --input audio/clip.wav \
  --limit 0 --peak-red 0.5
```

Writes the processed WAV next to the input.

## EGFxSet effect pipeline

`scripts/egfx_pipeline.py` runs the full download → align → train → export →
scaffold-plugin → commit → push loop for any effect in [EGFxSet (Zenodo 7044411)](https://zenodo.org/records/7044411):

```
uv run python scripts/egfx_pipeline.py BluesDriver
uv run python scripts/egfx_pipeline.py BluesDriver --no-push   # stop before push
uv run python scripts/egfx_pipeline.py BluesDriver --no-commit # just train + scaffold
```

Each run is idempotent — existing datasets, caches, artifacts, and plugin
dirs are reused. Supported effects are listed in the `EFFECTS` dict at the
top of the script; add a new entry (slug, display name, zip filename, CLAP
feature keyword) to run one that isn't there yet.

All captures are at a single knob setting, so the trained models have
`nparams = 0` and the generated plugins expose no knobs — see
[plugins/README.md](plugins/README.md#special-case-nparams--0).

### Trained models

Direct-arch TCN, 4 blocks · k=13 · dilation-growth=10 · channel-width=32,
20k steps at bf16 / bs=16 / lr=1e-3. Receptive field = **13 333 samples
(~278 ms at 48 kHz)** — enough for short-tail distortions and modulations;
stretched thin for long-decay reverbs and delays.

| Effect           | Pedal modeled         | val_loss | Notes                                                     |
| ---------------- | --------------------- | -------: | --------------------------------------------------------- |
| TubeScreamer     | Ibanez Mini           |   0.9398 | Reference run (pre-pipeline)                              |
| **BluesDriver**  | Boss BD-2             |   0.9617 | Clean distortion fit, on par with TS                      |
| **RAT**          | ProCo RAT2            |   1.0256 | Distortion, slightly harder                               |
| **Phaser**       | MXR Phase 45          |   1.0327 | Modulation tracks well (tiny L1, STFT dominates)          |
| **Spring-Reverb**| Orange CR-60 (spring) |   1.1367 | Better than expected given the 278 ms RF                  |
| **Chorus**       | Boss CE-3             |   1.2025 | Pitch + delay modulation is harder                        |
| **Flanger**      | Mooer E-Lady          |   1.7031 | Fast comb filtering stretches the RF — weakest fit        |

Lower val_loss = closer fit (L1 + MR-STFT magnitude). Distortions
(BluesDriver, RAT, TubeScreamer) are the most convincing; modulation effects
are reasonable approximations; Flanger is the one to listen-test critically —
comb-filter sweeps likely want a deeper/longer-RF architecture.

## Citation

```
@inproceedings{steinmetz2022efficient,
    title={Efficient neural networks for real-time modeling of analog dynamic range compression},
    author={Steinmetz, Christian J. and Reiss, Joshua D.},
    booktitle={152nd AES Convention},
    year={2022}}
```

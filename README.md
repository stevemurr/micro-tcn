<div align="center">

# micro-TCN
| [Paper](https://arxiv.org/abs/2102.06200) | [Demo](https://csteinmetz1.github.io/tcn-audio-effects/) |

Efficient neural networks for real-time modeling of analog dynamic range compression.

</div>

<div align="center">
<img src="plots/tcn-arch.svg">
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
  --batch-size 16 --lr 1e-3 \
  --max-steps 20000 --warmup-steps 500 --eval-every 1000 \
  --precision bf16 --num-workers 4
```

First run decodes every WAV into `<root-dir>/.cache/{subset}_{input,target}.bin` (one-time, ~14 GB). Later runs mmap it directly. Training is step-based: linear warmup over `--warmup-steps`, cosine decay to 0 by `--max-steps`. Validation runs every `--eval-every` steps; pass `--val-max-batches N` to cap val batches per eval for faster feedback. Per-eval train/val losses are written to `<artifact-dir>/log.csv`; top-k and `last.ckpt` land in `<artifact-dir>/checkpoints/`.

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

## Citation

```
@inproceedings{steinmetz2022efficient,
    title={Efficient neural networks for real-time modeling of analog dynamic range compression},
    author={Steinmetz, Christian J. and Reiss, Joshua D.},
    booktitle={152nd AES Convention},
    year={2022}}
```

"""Checkpoint evaluation over a dataset subset, with spectral + time-domain metrics."""
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from microtcn.data import load_dataset
from microtcn.metrics import all_metrics
from microtcn.model import TCN
from microtcn.utils import causal_crop, center_crop


METRIC_KEYS = ("stft_l1", "log_stft_l1", "mrstft_l1", "rms_env_l1", "si_sdr_db", "centroid_err_hz")


def load_tcn(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TCN(
        nparams=cfg.get("nparams", 2),
        nblocks=cfg["nblocks"],
        dilation_growth=cfg["dilation_growth"],
        kernel_size=cfg["kernel_size"],
        channel_width=cfg["channel_width"],
        causal=cfg["causal"],
        arch=cfg.get("arch", "direct"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def evaluate(
    root_dir: str,
    checkpoint_path: str,
    subset: str = "val",
    eval_length: int = 131072,
    batch_size: int = 8,
    num_workers: int = 4,
    max_batches: int | None = None,
    save_json: str | None = None,
    loader: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_tcn(checkpoint_path, device)
    crop_fn = causal_crop if cfg["causal"] else center_crop
    sample_rate = cfg.get("sample_rate", 44100)

    dataset = load_dataset(root_dir, subset=subset, length=eval_length, loader=loader)
    nparams = getattr(dataset, "nparams", 2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    per_class: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {k: [] for k in METRIC_KEYS}
    )
    overall = {k: [] for k in METRIC_KEYS}

    with torch.no_grad():
        for batch_idx, (x, target, params) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device); target = target.to(device); params = params.to(device)
            pred = model(x, params)
            target_c = crop_fn(target, pred.shape[-1])

            for i in range(pred.shape[0]):
                p = pred[i:i+1]
                t = target_c[i:i+1]
                m = all_metrics(p, t, sample_rate=sample_rate)
                if nparams >= 2:
                    key = f"{int(params[i, 0, 0].item())}-{int(params[i, 0, 1].item() * 100):03d}"
                else:
                    key = "all"
                for mk, mv in m.items():
                    per_class[key][mk].append(mv)
                    overall[mk].append(mv)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"checkpoint: {checkpoint_path}")
    print(f"arch: {cfg.get('arch', 'direct')}  subset: {subset}  examples: "
          f"{sum(len(v['stft_l1']) for v in per_class.values())}")
    print()
    header = f"{'class':>8}  " + "  ".join(f"{k:>14}" for k in METRIC_KEYS)
    print(header)
    print("-" * len(header))
    for key in sorted(per_class):
        v = per_class[key]
        row = f"{key:>8}  " + "  ".join(f"{_mean(v[k]):>14.4f}" for k in METRIC_KEYS)
        print(row)
    print("-" * len(header))
    mean_row = f"{'mean':>8}  " + "  ".join(f"{_mean(overall[k]):>14.4f}" for k in METRIC_KEYS)
    print(mean_row)

    result = {
        "checkpoint": checkpoint_path,
        "arch": cfg.get("arch", "direct"),
        "subset": subset,
        "sample_rate": sample_rate,
        "per_class": {k: {mk: _mean(mv) for mk, mv in v.items()} for k, v in per_class.items()},
        "overall": {k: _mean(overall[k]) for k in METRIC_KEYS},
    }
    if save_json is not None:
        with open(save_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {save_json}")
    return result

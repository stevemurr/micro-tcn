"""Checkpoint evaluation over a dataset subset, per-param-class."""
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from microtcn.data import SignalTrainLA2ADataset
from microtcn.loss import L1STFT
from microtcn.model import TCN
from microtcn.utils import causal_crop, center_crop


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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_tcn(checkpoint_path, device)
    crop_fn = causal_crop if cfg["causal"] else center_crop
    loss_fn = L1STFT().to(device)

    dataset = SignalTrainLA2ADataset(root_dir, subset=subset, length=eval_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    per_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"l1": [], "stft": [], "agg": []})
    with torch.no_grad():
        for x, target, params in loader:
            x = x.to(device); target = target.to(device); params = params.to(device)
            pred = model(x, params)
            target_c = crop_fn(target, pred.shape[-1])
            for i in range(pred.shape[0]):
                agg, parts = loss_fn(pred[i:i+1], target_c[i:i+1])
                key = f"{int(params[i, 0, 0].item())}-{int(params[i, 0, 1].item() * 100):03d}"
                per_class[key]["l1"].append(parts["l1"].item())
                per_class[key]["stft"].append(parts["stft"].item())
                per_class[key]["agg"].append(agg.item())

    print(f"checkpoint: {checkpoint_path}")
    print(f"subset: {subset}  examples: {len(dataset)}")
    print(f"{'class':>8}  {'L1':>8}  {'STFT':>8}  {'agg':>8}")
    l1_all, stft_all, agg_all = [], [], []
    for key in sorted(per_class):
        v = per_class[key]
        print(f"{key:>8}  {sum(v['l1'])/len(v['l1']):8.4f}  "
              f"{sum(v['stft'])/len(v['stft']):8.4f}  {sum(v['agg'])/len(v['agg']):8.4f}")
        l1_all += v['l1']; stft_all += v['stft']; agg_all += v['agg']
    print(f"{'mean':>8}  {sum(l1_all)/len(l1_all):8.4f}  "
          f"{sum(stft_all)/len(stft_all):8.4f}  {sum(agg_all)/len(agg_all):8.4f}")
    return per_class

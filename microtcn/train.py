"""Raw PyTorch training loop."""
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from microtcn.data import SignalTrainLA2ADataset
from microtcn.loss import L1STFT
from microtcn.model import TCN
from microtcn.utils import causal_crop, center_crop


_AMP_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _forward_loss(model, loss_fn, x, target, params, crop_fn, amp_dtype, device):
    if amp_dtype is not None and device.type == "cuda":
        with torch.autocast("cuda", dtype=amp_dtype):
            pred = model(x, params)
            target = crop_fn(target, pred.shape[-1])
            return loss_fn(pred, target.to(pred.dtype))
    pred = model(x, params)
    target = crop_fn(target, pred.shape[-1])
    return loss_fn(pred, target)


def run_training(
    root_dir: str,
    artifact_dir: str,
    nblocks: int = 4,
    dilation_growth: int = 10,
    kernel_size: int = 13,
    channel_width: int = 32,
    causal: bool = True,
    train_fraction: float = 0.1,
    train_length: int = 65536,
    eval_length: int = 131072,
    batch_size: int = 16,
    val_batch_size: int = 8,
    lr: float = 1e-3,
    max_epochs: int = 60,
    precision: str = "bf16",
    num_workers: int = 4,
    save_top_k: int = 3,
    seed: int = 42,
    log_every: int = 50,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _AMP_DTYPES[precision]

    out_dir = Path(artifact_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SignalTrainLA2ADataset(root_dir, subset="train", length=train_length, fraction=train_fraction)
    val_ds = SignalTrainLA2ADataset(root_dir, subset="val", length=eval_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    model = TCN(
        nblocks=nblocks, dilation_growth=dilation_growth,
        kernel_size=kernel_size, channel_width=channel_width, causal=causal,
    ).to(device)
    print(model)
    print(f"receptive field: {model.receptive_field()} samples "
          f"({model.receptive_field() / train_ds.sample_rate * 1000:.1f} ms)")

    loss_fn = L1STFT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    cfg = dict(
        nblocks=nblocks, dilation_growth=dilation_growth, kernel_size=kernel_size,
        channel_width=channel_width, causal=causal, nparams=2,
        sample_rate=train_ds.sample_rate,
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_l1", "val_stft", "lr"])

    crop_fn = causal_crop if causal else center_crop
    best: list[tuple[float, Path]] = []

    for epoch in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        for step, (x, target, params) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            params = params.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss, _ = _forward_loss(model, loss_fn, x, target, params, crop_fn, amp_dtype, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if step % log_every == 0:
                print(f"epoch {epoch:02d} step {step:04d}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        train_loss = sum(train_losses) / max(len(train_losses), 1)

        model.eval()
        val_agg, val_l1, val_stft = [], [], []
        with torch.no_grad():
            for x, target, params in val_loader:
                x = x.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                params = params.to(device, non_blocking=True)
                total, parts = _forward_loss(model, loss_fn, x, target, params, crop_fn, amp_dtype, device)
                val_agg.append(total.item())
                val_l1.append(parts["l1"].item())
                val_stft.append(parts["stft"].item())
        val_loss = sum(val_agg) / max(len(val_agg), 1)
        val_l1_mean = sum(val_l1) / max(len(val_l1), 1)
        val_stft_mean = sum(val_stft) / max(len(val_stft), 1)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[epoch {epoch:02d}] train={train_loss:.4f}  "
            f"val={val_loss:.4f} (l1={val_l1_mean:.4f} stft={val_stft_mean:.4f})  lr={current_lr:.2e}",
            flush=True,
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_l1_mean, val_stft_mean, current_lr])

        ckpt_path = ckpt_dir / f"epoch={epoch:02d}-val={val_loss:.4f}.ckpt"
        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_loss": val_loss}
        torch.save(state, ckpt_path)
        torch.save(state, ckpt_dir / "last.ckpt")

        best.append((val_loss, ckpt_path))
        best.sort(key=lambda t: t[0])
        for _, p in best[save_top_k:]:
            if p.exists():
                p.unlink()
        best = best[:save_top_k]

    print(f"done. best val_loss={best[0][0]:.4f} at {best[0][1]}")
    return best[0][1]

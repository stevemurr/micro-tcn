"""Step-based training loop."""
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from microtcn.data import SignalTrainLA2ADataset
from microtcn.loss import L1STFT
from microtcn.model import TCN
from microtcn.utils import causal_crop, center_crop


_AMP_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _lr_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _forward(model, loss_fn, x, target, params, crop_fn, amp_dtype):
    if amp_dtype is not None and x.is_cuda:
        with torch.autocast("cuda", dtype=amp_dtype):
            pred = model(x, params)
            target = crop_fn(target, pred.shape[-1])
            return loss_fn(pred, target.to(pred.dtype))
    pred = model(x, params)
    target = crop_fn(target, pred.shape[-1])
    return loss_fn(pred, target)


@torch.no_grad()
def _validate(model, val_loader, loss_fn, crop_fn, amp_dtype, device, max_batches):
    model.eval()
    agg, l1s, stfts = [], [], []
    for i, (x, target, params) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)
        total, parts = _forward(model, loss_fn, x, target, params, crop_fn, amp_dtype)
        agg.append(total.item())
        l1s.append(parts["l1"].item())
        stfts.append(parts["stft"].item())
    model.train()
    n = max(len(agg), 1)
    return sum(agg) / n, sum(l1s) / n, sum(stfts) / n


def run_training(
    root_dir: str,
    artifact_dir: str,
    nblocks: int = 4,
    dilation_growth: int = 10,
    kernel_size: int = 13,
    channel_width: int = 32,
    causal: bool = True,
    arch: str = "direct",
    train_length: int = 65536,
    eval_length: int = 131072,
    batch_size: int = 16,
    val_batch_size: int = 8,
    lr: float = 1e-3,
    max_steps: int = 20000,
    warmup_steps: int = 500,
    eval_every: int = 1000,
    log_every: int = 50,
    val_max_batches: int | None = None,
    precision: str = "bf16",
    num_workers: int = 4,
    save_top_k: int = 3,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _AMP_DTYPES[precision]

    out_dir = Path(artifact_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SignalTrainLA2ADataset(root_dir, subset="train", length=train_length)
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
        arch=arch,
    ).to(device)
    print(model)
    print(f"arch: {arch}  "
          f"receptive field: {model.receptive_field()} samples "
          f"({model.receptive_field() / train_ds.sample_rate * 1000:.1f} ms)")
    print(f"train examples: {len(train_ds)}  val examples: {len(val_ds)}")

    loss_fn = L1STFT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_schedule(s, warmup_steps, max_steps),
    )

    cfg = dict(
        nblocks=nblocks, dilation_growth=dilation_growth, kernel_size=kernel_size,
        channel_width=channel_width, causal=causal, nparams=2, arch=arch,
        sample_rate=train_ds.sample_rate,
    )
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_loss", "val_l1", "val_stft", "lr", "alpha"])

    crop_fn = causal_crop if causal else center_crop
    best: list[tuple[float, Path]] = []
    train_iter = _infinite(train_loader)
    model.train()
    recent_losses: list[float] = []

    for step in range(1, max_steps + 1):
        x, target, params = next(train_iter)
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = _forward(model, loss_fn, x, target, params, crop_fn, amp_dtype)
        loss.backward()
        optimizer.step()
        scheduler.step()
        recent_losses.append(loss.item())

        if step % log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:06d}/{max_steps}  loss={loss.item():.4f}  lr={current_lr:.2e}",
                flush=True,
            )

        if step % eval_every == 0 or step == max_steps:
            val_loss, val_l1, val_stft = _validate(
                model, val_loader, loss_fn, crop_fn, amp_dtype, device, val_max_batches,
            )
            train_loss = sum(recent_losses) / max(len(recent_losses), 1)
            recent_losses.clear()
            current_lr = optimizer.param_groups[0]["lr"]

            alpha_val = model.alpha.item() if hasattr(model, "alpha") else 0.0
            extra = f"  α={alpha_val:.4f}" if arch == "hybrid" else ""
            print(
                f"[step {step:06d}] train={train_loss:.4f}  "
                f"val={val_loss:.4f} (l1={val_l1:.4f} stft={val_stft:.4f})  lr={current_lr:.2e}{extra}",
                flush=True,
            )
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, train_loss, val_loss, val_l1, val_stft, current_lr, alpha_val])

            ckpt_path = ckpt_dir / f"step={step:06d}-val={val_loss:.4f}.ckpt"
            state = {"model": model.state_dict(), "config": cfg, "step": step, "val_loss": val_loss}
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

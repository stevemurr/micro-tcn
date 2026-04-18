"""Export a direct-arch checkpoint to a JSON weights file for the CLAP plugin.

The output format is a plain JSON dump of the weights and fixed hyperparameters.
It's consumable by either:
  - RTNeural (via a custom C++ loader that assembles the graph from named layers)
  - a hand-rolled Rust inference runtime in plugins/tcn-clap

Hybrid-arch checkpoints are rejected — their sigmoid gain + residual branch
requires extending the plugin's inference runtime (not impossible, just not done).
"""
import json
from pathlib import Path

import torch

from microtcn.model import TCN


EXPORT_VERSION = 1


def _linear(layer: torch.nn.Linear) -> dict:
    return {
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "weight": layer.weight.detach().cpu().tolist(),          # (out, in)
        "bias": layer.bias.detach().cpu().tolist() if layer.bias is not None else None,
    }


def _conv1d(layer: torch.nn.Conv1d) -> dict:
    return {
        "in_channels": layer.in_channels,
        "out_channels": layer.out_channels,
        "kernel_size": layer.kernel_size[0],
        "dilation": layer.dilation[0],
        "groups": layer.groups,
        "weight": layer.weight.detach().cpu().tolist(),          # (out, in/groups, kernel)
        "bias": layer.bias.detach().cpu().tolist() if layer.bias is not None else None,
    }


def _bn1d(layer: torch.nn.BatchNorm1d) -> dict:
    return {
        "num_features": layer.num_features,
        "affine": layer.affine,
        "eps": layer.eps,
        "running_mean": layer.running_mean.detach().cpu().tolist(),
        "running_var": layer.running_var.detach().cpu().tolist(),
    }


def _prelu(layer: torch.nn.PReLU) -> dict:
    return {
        "num_parameters": layer.num_parameters,
        "weight": layer.weight.detach().cpu().tolist(),
    }


def export_direct(checkpoint_path: str, output_path: str) -> Path:
    """Write a JSON weights dump for a direct-arch checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    arch = cfg.get("arch", "direct")
    if arch != "direct":
        raise ValueError(
            f"export only supports arch='direct' at the moment (got {arch!r}). "
            "Hybrid export would require shipping a sigmoid gain + learned α in the plugin runtime."
        )

    model = TCN(
        nparams=cfg.get("nparams", 2),
        nblocks=cfg["nblocks"],
        dilation_growth=cfg["dilation_growth"],
        kernel_size=cfg["kernel_size"],
        channel_width=cfg["channel_width"],
        causal=cfg["causal"],
        arch="direct",
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    gen_linears = [m for m in model.gen if isinstance(m, torch.nn.Linear)]

    blocks_out = []
    for block in model.blocks:
        blocks_out.append({
            "conv1": _conv1d(block.conv1),
            "bn": _bn1d(block.film.bn),
            "adaptor": _linear(block.film.adaptor),
            "prelu": _prelu(block.relu),
            "res": _conv1d(block.res),
            "causal": block.causal,
        })

    payload = {
        "version": EXPORT_VERSION,
        "arch": "direct",
        "config": {
            "nblocks": cfg["nblocks"],
            "kernel_size": cfg["kernel_size"],
            "dilation_growth": cfg["dilation_growth"],
            "channel_width": cfg["channel_width"],
            "causal": cfg["causal"],
            "nparams": cfg.get("nparams", 2),
            "sample_rate": cfg.get("sample_rate", 44100),
            "receptive_field": model.receptive_field(),
        },
        "gen": [_linear(l) for l in gen_linears],
        "blocks": blocks_out,
        "output": _conv1d(model.output),
        "meta": {
            "source_checkpoint": str(Path(checkpoint_path).resolve()),
            "val_loss": ckpt.get("val_loss"),
            "step": ckpt.get("step"),
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)

    rf = model.receptive_field()
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"receptive field: {rf} samples "
          f"({rf / cfg.get('sample_rate', 44100) * 1000:.1f} ms at SR={cfg.get('sample_rate', 44100)})")
    return out_path

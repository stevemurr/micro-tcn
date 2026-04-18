"""Typer CLI for train / eval / comp."""
from typing import Optional

import typer

from microtcn.comp import compress
from microtcn.eval import evaluate
from microtcn.train import run_training

app = typer.Typer(
    name="microtcn",
    help="Efficient neural networks for real-time modeling of analog dynamic range compression.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("train")
def train_cmd(
    root_dir: str = typer.Option(..., help="SignalTrain dataset root."),
    artifact_dir: str = typer.Option(..., help="Where to write checkpoints and log.csv."),
    nblocks: int = typer.Option(4),
    dilation_growth: int = typer.Option(10),
    kernel_size: int = typer.Option(13),
    channel_width: int = typer.Option(32),
    causal: bool = typer.Option(True),
    arch: str = typer.Option("direct", help="'direct' (tanh head) | 'hybrid' (gain + coloration)."),
    train_length: int = typer.Option(65536),
    eval_length: int = typer.Option(131072),
    batch_size: int = typer.Option(16),
    val_batch_size: int = typer.Option(8),
    lr: float = typer.Option(1e-3),
    max_steps: int = typer.Option(20000, help="Total gradient updates."),
    warmup_steps: int = typer.Option(500, help="Linear warmup before cosine decay."),
    eval_every: int = typer.Option(1000, help="Run validation every N steps."),
    log_every: int = typer.Option(50),
    val_max_batches: Optional[int] = typer.Option(None, help="Cap val batches per eval (None = all)."),
    precision: str = typer.Option("bf16", help="'fp32' | 'bf16' | 'fp16'"),
    num_workers: int = typer.Option(4),
    save_top_k: int = typer.Option(3),
    seed: int = typer.Option(42),
):
    """Train a TCN model."""
    run_training(
        root_dir=root_dir, artifact_dir=artifact_dir,
        nblocks=nblocks, dilation_growth=dilation_growth, kernel_size=kernel_size,
        channel_width=channel_width, causal=causal, arch=arch,
        train_length=train_length, eval_length=eval_length,
        batch_size=batch_size, val_batch_size=val_batch_size, lr=lr,
        max_steps=max_steps, warmup_steps=warmup_steps, eval_every=eval_every,
        log_every=log_every, val_max_batches=val_max_batches,
        precision=precision, num_workers=num_workers,
        save_top_k=save_top_k, seed=seed,
    )


@app.command("eval")
def eval_cmd(
    root_dir: str = typer.Option(...),
    checkpoint: str = typer.Option(..., help="Path to a .ckpt file."),
    subset: str = typer.Option("val"),
    eval_length: int = typer.Option(131072),
    batch_size: int = typer.Option(8),
    num_workers: int = typer.Option(4),
    max_batches: Optional[int] = typer.Option(None, help="Cap val batches (None = all)."),
    save_json: Optional[str] = typer.Option(None, help="Dump per-class + overall metrics JSON."),
):
    """Evaluate a checkpoint with spectral + time-domain metrics, per-param-class."""
    evaluate(
        root_dir=root_dir, checkpoint_path=checkpoint, subset=subset,
        eval_length=eval_length, batch_size=batch_size, num_workers=num_workers,
        max_batches=max_batches, save_json=save_json,
    )


@app.command("comp")
def comp_cmd(
    checkpoint: str = typer.Option(..., help="Path to .ckpt."),
    input: str = typer.Option(..., "--input", "-i", help="WAV file to process."),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    limit: int = typer.Option(0, help="0=compress, 1=limit."),
    peak_red: float = typer.Option(0.5, help="Peak reduction 0..1."),
    device: str = typer.Option("cuda"),
):
    """Run a trained TCN over a WAV."""
    compress(
        checkpoint_path=checkpoint, input_path=input, output_path=output,
        limit=limit, peak_red=peak_red, device=device,
    )


if __name__ == "__main__":
    app()

import glob
import itertools
import os
import sys
import time
from typing import Optional

import torch
import typer
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from microtcn.lstm import LSTMModel
from microtcn.tcn import TCNModel


def _load_model(model_dir: str, model_id: str, gpu: bool = False):
    checkpoint_path = glob.glob(
        os.path.join(model_dir, model_id, "lightning_logs", "version_0", "checkpoints", "*")
    )[0]
    model_type = os.path.basename(model_id).split("-")[1]
    map_location = "cuda:0" if gpu else "cpu"

    if model_type == "LSTM":
        return LSTMModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, map_location=map_location
        )
    return TCNModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location=map_location
    )


def _get_files(input_path: str) -> list:
    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*"))
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        raise typer.BadParameter(f"'{input_path}' is not a valid file or directory")
    print(f"Found {len(files)} input file(s).")
    return files


def _process(model, inputfile: str, limit: float, peak_red: float, gpu: bool, verbose: bool):
    dec = AudioDecoder(inputfile)
    sr = dec.metadata.sample_rate
    input_ = dec.get_all_samples().data

    if input_.size(0) > 1:
        print(f"Warning: model is mono; downmixing {input_.size(0)} channels.")
        input_ = torch.sum(input_, dim=0)

    if sr != 44100:
        print(f"Warning: model operates at 44.1 kHz, got {sr} Hz.")

    params = torch.tensor([limit, peak_red])
    input_ = input_.view(1, 1, -1)
    params = params.view(1, 1, 2)

    if gpu:
        input_ = input_.to("cuda:0")
        params = params.to("cuda:0")
        model.to("cuda:0")

    tic = time.perf_counter()
    out = model(input_, params).view(1, -1)
    toc = time.perf_counter()

    if verbose:
        duration = input_.size(-1) / 44100
        elapsed = toc - tic
        print(
            f"Processed {duration:0.2f} sec in {elapsed:0.3f} sec => {duration / elapsed:0.1f}x real-time"
        )

    srcpath = os.path.dirname(inputfile)
    srcbasename = os.path.basename(inputfile).split(".")[0]
    outfile = os.path.join(srcpath, srcbasename)
    outfile += f"-{limit:1.0f}-{int(peak_red * 100)}-tcn-300-c.wav"
    AudioEncoder(out.cpu(), sample_rate=44100).to_file(outfile)


def comp(
    input: Optional[str] = typer.Option(None, "--input", "-i", help="File or folder to process."),
    model_dir: str = typer.Option("./lightning_logs/bulk"),
    model_id: str = typer.Option("1-uTCN-300__causal__4-10-13__fraction-0.01-bs32"),
    list_models: bool = typer.Option(False, help="Print the available models and exit."),
    gpu: bool = typer.Option(False),
    verbose: bool = typer.Option(False),
    limit: int = typer.Option(0, help="Compressor mode: 'limit' (1) or 'compress' (0)."),
    peak_red: float = typer.Option(0.5, help="Peak reduction, 0..1."),
    full: bool = typer.Option(False, help="Sweep the full compressor parameter space."),
):
    """Run a trained compressor model over audio files."""
    if list_models:
        models = sorted(glob.glob(os.path.join(model_dir, "*")))
        print(f"Found {len(models)} models in {model_dir}")
        for m in models:
            print(os.path.basename(m))
        sys.exit(0)

    if input is None:
        raise typer.BadParameter("--input is required unless --list-models is set")

    model = _load_model(model_dir, model_id, gpu=gpu)
    files = _get_files(input)
    for f in files:
        if full:
            limits = [0.0, 1.0]
            thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for lim, thr in itertools.product(limits, thresholds):
                _process(model, f, lim, thr, gpu=gpu, verbose=verbose)
        else:
            _process(model, f, float(limit), peak_red, gpu=gpu, verbose=verbose)

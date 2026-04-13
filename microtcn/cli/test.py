import glob
import os
import pickle
import sys
from typing import Optional

import auraloss
import numpy as np
import pyloudnorm as pyln
import pytorch_lightning as pl
import torch
import torchaudio
import typer

from microtcn.data import SignalTrainLA2ADataset
from microtcn.lstm import LSTMModel
from microtcn.tcn import TCNModel
from microtcn.utils import causal_crop, center_crop


def test(
    root_dir: str = typer.Option("./data"),
    model_dir: str = typer.Option("./lightning_logs/bulk"),
    save_dir: Optional[str] = typer.Option(None, help="If set, render output/input/target wavs here."),
    preload: bool = typer.Option(False),
    half: bool = typer.Option(False),
    fast: bool = typer.Option(False, help="Skip LSTM models."),
    sample_rate: int = typer.Option(44100),
    eval_subset: str = typer.Option("val"),
    eval_length: int = typer.Option(8388608),
    batch_size: int = typer.Option(1),
    num_workers: int = typer.Option(32),
):
    """Evaluate all trained models under ``model_dir`` on the eval subset."""
    pl.seed_everything(42)

    test_dataset = SignalTrainLA2ADataset(
        root_dir,
        subset=eval_subset,
        half=False,
        preload=preload,
        length=eval_length,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    overall_results: dict = {}

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    l1 = torch.nn.L1Loss()
    stft = auraloss.freq.STFTLoss()
    meter = pyln.Meter(sample_rate)

    models = sorted(glob.glob(os.path.join(model_dir, "*")))

    for idx, md in enumerate(models):
        results: dict = {}

        checkpoint_path = glob.glob(
            os.path.join(md, "lightning_logs", "version_0", "checkpoints", "*")
        )[0]

        model_id = os.path.basename(md)
        batch_size_id = int(model_id.split("-")[-1][2:])
        model_type = model_id.split("-")[1]
        epoch = int(os.path.basename(checkpoint_path).split("-")[0].split("=")[-1])

        if model_type == "LSTM":
            if fast:
                continue
            model = LSTMModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path, map_location="cuda:0"
            )
        else:
            model = TCNModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path, map_location="cuda:0"
            )

        print(
            f" {idx + 1}/{len(models)} : epoch: {epoch} {model_id} batch size {batch_size_id}"
        )

        model.cuda()
        model.eval()
        if half:
            model.half()

        pl.seed_everything(42)

        for bidx, batch in enumerate(test_dataloader):
            sys.stdout.write(f" Evaluating {bidx}/{len(test_dataloader)}...\r")
            sys.stdout.flush()

            input_, target, params = batch
            input_ = input_.to("cuda:0")
            target = target.to("cuda:0")
            params = params.to("cuda:0")

            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(input_, params)

                if model.hparams.causal:
                    input_crop = causal_crop(input_, output.shape[-1])
                    target_crop = causal_crop(target, output.shape[-1])
                else:
                    input_crop = center_crop(input_, output.shape[-1])
                    target_crop = center_crop(target, output.shape[-1])

            for _, (i, o, t, p) in enumerate(
                zip(
                    torch.split(input_crop, 1, dim=0),
                    torch.split(output, 1, dim=0),
                    torch.split(target_crop, 1, dim=0),
                    torch.split(params, 1, dim=0),
                )
            ):
                l1_loss = l1(o, t).cpu().numpy()
                stft_loss = stft(o, t).cpu().numpy()
                aggregate_loss = l1_loss + stft_loss

                target_lufs = meter.integrated_loudness(t.squeeze().cpu().numpy())
                output_lufs = meter.integrated_loudness(o.squeeze().cpu().numpy())
                l1_lufs = np.abs(output_lufs - target_lufs)

                l1i_loss = (l1(i, t) - l1(o, t)).cpu().numpy()
                stfti_loss = (stft(i, t) - stft(o, t)).cpu().numpy()

                params_np = p.squeeze().cpu().numpy()
                params_key = f"{params_np[0]:1.0f}-{params_np[1] * 100:03.0f}"

                if save_dir is not None:
                    ofile = os.path.join(save_dir, f"{params_key}-{bidx}-output--{model_id}.wav")
                    ifile = os.path.join(save_dir, f"{params_key}-{bidx}-input.wav")
                    tfile = os.path.join(save_dir, f"{params_key}-{bidx}-target.wav")
                    torchaudio.save(ofile, o.view(1, -1).cpu().float(), sample_rate)
                    if not os.path.isfile(ifile):
                        torchaudio.save(ifile, i.view(1, -1).cpu().float(), sample_rate)
                    if not os.path.isfile(tfile):
                        torchaudio.save(tfile, t.view(1, -1).cpu().float(), sample_rate)

                entry = results.setdefault(
                    params_key,
                    {"L1": [], "L1i": [], "STFT": [], "STFTi": [], "LUFS": [], "Agg": []},
                )
                entry["L1"].append(l1_loss)
                entry["L1i"].append(l1i_loss)
                entry["STFT"].append(stft_loss)
                entry["STFTi"].append(stfti_loss)
                entry["LUFS"].append(l1_lufs)
                entry["Agg"].append(aggregate_loss)

        l1_scores, lufs_scores, stft_scores, agg_scores = [], [], [], []
        print("-" * 64)
        print("Config      L1         STFT      LUFS")
        print("-" * 64)
        for key, val in results.items():
            print(
                f"{key}    {np.mean(val['L1']):0.2e}    {np.mean(val['STFT']):0.3f}       {np.mean(val['LUFS']):0.3f}"
            )
            l1_scores += val["L1"]
            stft_scores += val["STFT"]
            lufs_scores += val["LUFS"]
            agg_scores += val["Agg"]

        print("-" * 64)
        print(
            f"Mean     {np.mean(l1_scores):0.2e}    {np.mean(stft_scores):0.3f}      {np.mean(lufs_scores):0.3f}"
        )
        print()
        overall_results[model_id] = results

    with open(f"test_results_{eval_subset}.p", "wb") as f:
        pickle.dump(overall_results, f)

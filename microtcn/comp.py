"""Process a WAV through a trained TCN checkpoint."""
import time
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from microtcn.eval import load_tcn


def compress(
    checkpoint_path: str,
    input_path: str,
    output_path: str | None = None,
    limit: int = 0,
    peak_red: float = 0.5,
    device: str = "cuda",
    verbose: bool = True,
) -> Path:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, cfg = load_tcn(checkpoint_path, dev)
    sr = cfg.get("sample_rate", 44100)

    dec = AudioDecoder(input_path)
    audio = dec.get_all_samples().data
    if audio.shape[0] > 1:
        print(f"Warning: mono model; downmixing {audio.shape[0]} channels.")
        audio = audio.mean(dim=0, keepdim=True)
    file_sr = dec.metadata.sample_rate
    if file_sr != sr:
        print(f"Warning: model SR={sr}, file SR={file_sr}")

    x = audio.unsqueeze(0).to(dev).float()
    params = torch.tensor([[[float(limit), float(peak_red)]]], device=dev)

    tic = time.perf_counter()
    with torch.no_grad():
        y = model(x, params).squeeze(0).cpu().float()
    toc = time.perf_counter()

    if verbose:
        dur = audio.shape[-1] / sr
        print(f"Processed {dur:.2f}s in {toc - tic:.3f}s ({dur / (toc - tic):.1f}x realtime)")

    if output_path is None:
        p = Path(input_path)
        output_path = p.with_name(f"{p.stem}-l{int(limit)}-p{int(peak_red * 100):02d}.wav")
    output_path = Path(output_path)
    AudioEncoder(y.clamp(-1.0, 1.0), sample_rate=sr).to_file(str(output_path))
    print(f"wrote {output_path}")
    return output_path

"""End-to-end per-effect pipeline: download → cache → train → export → scaffold plugin.

Usage:
    uv run python scripts/egfx_pipeline.py BluesDriver
    uv run python scripts/egfx_pipeline.py BluesDriver --no-push

Each effect is defined in EFFECTS below. The script is idempotent: existing
datasets, caches, artifacts, and plugin dirs are reused.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path.home() / "jupyter-redux" / "datasets"
ARTIFACTS_DIR = Path.home() / "jupyter-redux" / "artifacts"
CLEAN_DIR = DATASETS_DIR / "Clean"

# CLAP feature keyword per effect drives the third feature string in lib.rs.
EFFECTS: dict[str, dict] = {
    "BluesDriver": {
        "slug": "bluesdriver",
        "display": "Blues Driver",
        "zenodo": "BluesDriver.zip",
        "dir": "BluesDriver",
        "feature": "distortion",
        "description": "Neural Boss BD-2 Blues Driver overdrive.",
    },
    "RAT": {
        "slug": "rat",
        "display": "RAT",
        "zenodo": "RAT.zip",
        "dir": "RAT",
        "feature": "distortion",
        "description": "Neural ProCo RAT2 distortion.",
    },
    "Phaser": {
        "slug": "phaser",
        "display": "Phaser",
        "zenodo": "Phaser.zip",
        "dir": "Phaser",
        "feature": "phaser",
        "description": "Neural MXR Phase 45 phaser.",
    },
    "Chorus": {
        "slug": "chorus",
        "display": "Chorus",
        "zenodo": "Chorus.zip",
        "dir": "Chorus",
        "feature": "chorus",
        "description": "Neural Boss CE-3 chorus.",
    },
    "Flanger": {
        "slug": "flanger",
        "display": "Flanger",
        "zenodo": "Flanger.zip",
        "dir": "Flanger",
        "feature": "flanger",
        "description": "Neural Mooer E-Lady flanger.",
    },
    "Spring-Reverb": {
        "slug": "spring-reverb",
        "display": "Spring Reverb",
        "zenodo": "Spring-Reverb.zip",
        "dir": "Spring-Reverb",
        "feature": "reverb",
        "description": "Neural Orange CR-60 spring reverb.",
    },
}


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd or REPO), check=True)


def ensure_dataset(effect: dict) -> Path:
    effect_dir = DATASETS_DIR / effect["dir"]
    if effect_dir.exists() and any(effect_dir.iterdir()):
        print(f"dataset present: {effect_dir}")
        return effect_dir

    url = f"https://zenodo.org/records/7044411/files/{effect['zenodo']}"
    zip_path = DATASETS_DIR / effect["zenodo"]
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        run(["curl", "-L", "--fail", "-o", zip_path, url])
    print(f"unzipping {zip_path} → {DATASETS_DIR}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATASETS_DIR)
    zip_path.unlink()
    return effect_dir


def build_cache(effect: dict, effect_dir: Path) -> Path:
    cache_dir = DATASETS_DIR / f".cache_{effect['slug']}"
    if (cache_dir / "train_input.bin").exists() and (cache_dir / "val_input.bin").exists():
        print(f"cache present: {cache_dir}")
        return cache_dir
    run([
        "uv", "run", "microtcn", "build-egfx",
        "--clean-dir", CLEAN_DIR,
        "--wet-dir", effect_dir,
        "--cache-dir", cache_dir,
    ])
    return cache_dir


def train(effect: dict, cache_dir: Path) -> Path:
    artifact_dir = ARTIFACTS_DIR / f"{effect['slug']}-direct-v1"
    last_ckpt = artifact_dir / "checkpoints" / "last.ckpt"
    if last_ckpt.exists():
        print(f"artifact present: {artifact_dir}")
        return artifact_dir

    run([
        "uv", "run", "microtcn", "train",
        "--root-dir", cache_dir,
        "--artifact-dir", artifact_dir,
        "--arch", "direct",
        "--causal",
        "--nblocks", "4",
        "--dilation-growth", "10",
        "--kernel-size", "13",
        "--channel-width", "32",
        "--batch-size", "16",
        "--lr", "1e-3",
        "--max-steps", "20000",
        "--warmup-steps", "500",
        "--eval-every", "1000",
        "--precision", "bf16",
        "--num-workers", "4",
        "--loader", "egfx",
    ])
    return artifact_dir


CKPT_VAL_RE = re.compile(r"val=(\d+\.\d+)\.ckpt$")


def best_ckpt(artifact_dir: Path) -> Path:
    ckpts = list((artifact_dir / "checkpoints").glob("step=*.ckpt"))
    if not ckpts:
        raise RuntimeError(f"no checkpoints found in {artifact_dir}")
    return min(ckpts, key=lambda p: float(CKPT_VAL_RE.search(p.name).group(1)))


def scaffold_plugin(effect: dict) -> Path:
    slug = effect["slug"]
    crate_dir = REPO / "plugins" / f"tcn-{slug}"
    template = REPO / "plugins" / "tcn-tubescreamer"
    if crate_dir.exists():
        print(f"plugin crate present: {crate_dir}")
    else:
        print(f"scaffolding {crate_dir} from {template}")
        shutil.copytree(template, crate_dir)
        # The copied tcn.json serves as a placeholder so `cargo check` / include_str!
        # works before training completes; export overwrites it with the real model.

    # Cargo.toml
    cargo_toml = crate_dir / "Cargo.toml"
    cargo_toml.write_text(
        cargo_toml.read_text()
        .replace("tcn-tubescreamer", f"tcn-{slug}")
        .replace(
            "Real-time neural tube-screamer distortion — micro-TCN inference.",
            f"{effect['description']} micro-TCN inference.",
        )
    )

    # lib.rs — identifiers, descriptor, feature keyword
    lib_rs = crate_dir / "src" / "lib.rs"
    text = lib_rs.read_text()
    text = text.replace(
        r'b"com.microtcn.tcn-tubescreamer\0"',
        f'b"com.microtcn.tcn-{slug}\\0"',
    )
    text = text.replace(
        r'b"micro-TCN TubeScreamer\0"',
        f'b"micro-TCN {effect["display"]}\\0"',
    )
    text = text.replace(
        r'b"Neural tube-screamer distortion.\0"',
        f'b"{effect["description"]}\\0"',
    )
    text = text.replace(
        r'b"distortion\0".as_ptr() as *const c_char,',
        f'b"{effect["feature"]}\\0".as_ptr() as *const c_char,',
    )
    lib_rs.write_text(text)
    return crate_dir


def export_json(effect: dict, artifact_dir: Path, crate_dir: Path) -> tuple[Path, Path]:
    best = best_ckpt(artifact_dir)
    out_path = crate_dir / "assets" / "tcn.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "uv", "run", "microtcn", "export",
        "--checkpoint", best,
        "--output", out_path,
    ])
    return out_path, best


def register_plugin(effect: dict) -> None:
    slug = effect["slug"]
    # plugins/Cargo.toml — add member
    ws = REPO / "plugins" / "Cargo.toml"
    txt = ws.read_text()
    member_line = f'    "tcn-{slug}",'
    if member_line not in txt:
        lines = txt.splitlines()
        idx = next(i for i, l in enumerate(lines) if l.strip() == '"tcn-tubescreamer",')
        lines.insert(idx + 1, member_line)
        ws.write_text("\n".join(lines) + "\n")

    # install-macos.sh — add to ALL_PLUGINS + env_var_for
    script = REPO / "plugins" / "install-macos.sh"
    s = script.read_text()
    if f"tcn-{slug}" not in s:
        s = re.sub(
            r"ALL_PLUGINS=\(([^)]+)\)",
            lambda m: f"ALL_PLUGINS=({m.group(1).strip()} tcn-{slug})",
            s,
        )
        env_var = f"TCN_{slug.upper().replace('-', '_')}_MODEL"
        case_line = f'        tcn-{slug}) echo "{env_var}" ;;'
        s = re.sub(
            r'(\n        tcn-tubescreamer\) echo "TCN_TUBESCREAMER_MODEL" ;;)',
            rf"\1\n{case_line}",
            s,
        )
        script.write_text(s)


def commit_push(effect: dict, best: Path, push: bool) -> None:
    val = float(CKPT_VAL_RE.search(best.name).group(1))
    slug = effect["slug"]
    msg = (
        f"Add tcn-{slug} CLAP plugin (val={val:.4f})\n\n"
        f"Trained direct-arch TCN on EGFxSet {effect['display']} — "
        f"20k steps, bf16, bs=16, lr=1e-3."
    )
    run(["git", "add", f"plugins/tcn-{slug}", "plugins/Cargo.toml", "plugins/install-macos.sh"])
    # --allow-empty-message not used; git will error if nothing staged.
    run(["git", "commit", "-m", msg])
    if push:
        run(["git", "push", "origin", "main"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("effect", choices=list(EFFECTS))
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args()

    effect = EFFECTS[args.effect]
    effect_dir = ensure_dataset(effect)
    cache_dir = build_cache(effect, effect_dir)
    artifact_dir = train(effect, cache_dir)
    crate_dir = scaffold_plugin(effect)
    _, best = export_json(effect, artifact_dir, crate_dir)
    register_plugin(effect)
    if not args.no_commit:
        commit_push(effect, best, push=not args.no_push)


if __name__ == "__main__":
    main()

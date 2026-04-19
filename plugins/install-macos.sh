#!/usr/bin/env bash
# Build + install micro-TCN plugins on macOS.
#
# Usage:
#   ./install-macos.sh                         # install every plugin in the workspace
#   ./install-macos.sh tcn-la2a                # install just one
#   ./install-macos.sh tcn-la2a tcn-tubescreamer
#
# Per-plugin runtime model overrides are read from env vars (see table below).
# Set them before invoking to drop a custom tcn.json into each plugin's
# Contents/Resources/. Absent variables leave the bundled default in place.
#
#   tcn-la2a         ← TCN_LA2A_MODEL=/path/to/la2a.json
#   tcn-tubescreamer ← TCN_TUBESCREAMER_MODEL=/path/to/ts.json

set -euo pipefail

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "install-macos.sh: macOS only (got OSTYPE=$OSTYPE)" >&2
    exit 1
fi

# Always run from plugins/ workspace root so the xtask alias resolves and the
# bundled output lands in plugins/target/bundled/.
cd "$(dirname "$0")"

# Every plugin crate (any workspace member with a cdylib and an assets/ dir)
# we know how to install. Add new plugins here once they gain a Cargo.toml.
ALL_PLUGINS=(tcn-la2a tcn-tubescreamer tcn-bluesdriver tcn-rat tcn-phaser tcn-chorus tcn-flanger)

# Each plugin declares which env var it reads for its runtime model override
# (via const MODEL_ENV_OVERRIDE in its lib.rs). Keep this table in sync.
env_var_for () {
    case "$1" in
        tcn-la2a)         echo "TCN_LA2A_MODEL" ;;
        tcn-tubescreamer) echo "TCN_TUBESCREAMER_MODEL" ;;
        tcn-flanger) echo "TCN_FLANGER_MODEL" ;;
        tcn-chorus) echo "TCN_CHORUS_MODEL" ;;
        tcn-phaser) echo "TCN_PHASER_MODEL" ;;
        tcn-rat) echo "TCN_RAT_MODEL" ;;
        tcn-bluesdriver) echo "TCN_BLUESDRIVER_MODEL" ;;
        *) echo "" ;;
    esac
}

if [ "$#" -eq 0 ]; then
    PLUGINS=("${ALL_PLUGINS[@]}")
else
    PLUGINS=("$@")
fi

# Validate up front so a typo doesn't half-install.
for p in "${PLUGINS[@]}"; do
    found=0
    for known in "${ALL_PLUGINS[@]}"; do
        if [ "$known" = "$p" ]; then found=1; break; fi
    done
    if [ "$found" -ne 1 ]; then
        echo "Unknown plugin: $p" >&2
        echo "Available: ${ALL_PLUGINS[*]}" >&2
        exit 1
    fi
done

BUILD_PROFILE="--release"
BUNDLED="target/bundled"
CLAP_DEST="$HOME/Library/Audio/Plug-Ins/CLAP"
VST3_DEST="$HOME/Library/Audio/Plug-Ins/VST3"

install_one_bundle () {
    local kind="$1"                              # clap or vst3
    local src="$2"                               # source bundle path
    local dest_dir="$3"                          # destination plugin directory
    local model_override="$4"                    # path or empty

    if [ ! -d "$src" ]; then
        echo "No $src — 'cargo xtask bundle' didn't produce $kind output." >&2
        return
    fi

    local bundle_name
    bundle_name="$(basename "$src")"
    local dest="$dest_dir/$bundle_name"

    mkdir -p "$dest_dir"
    rm -rf "$dest"
    cp -R "$src" "$dest"

    if [ -n "$model_override" ]; then
        if [ ! -f "$model_override" ]; then
            echo "Model override not found: $model_override" >&2
            exit 1
        fi
        mkdir -p "$dest/Contents/Resources"
        cp "$model_override" "$dest/Contents/Resources/tcn.json"
    fi

    # Re-codesign (ad-hoc) — mandatory whenever bundle contents change. Without
    # this macOS treats the bundle as tampered and silently refuses to load it.
    codesign -s - --force --deep "$dest"
    echo "installed $dest"
}

install_plugin () {
    local crate="$1"
    local env_var
    env_var="$(env_var_for "$crate")"
    local model_override=""
    if [ -n "$env_var" ]; then
        # Read the env var indirectly (${!var} is bash-ism we want here).
        model_override="${!env_var:-}"
    fi

    echo "==> Building $crate ($BUILD_PROFILE)"
    cargo xtask bundle "$crate" $BUILD_PROFILE

    echo "==> Installing $crate"
    install_one_bundle clap "$BUNDLED/${crate}.clap" "$CLAP_DEST" "$model_override"
    install_one_bundle vst3 "$BUNDLED/${crate}.vst3" "$VST3_DEST" "$model_override"

    if [ -n "$model_override" ]; then
        echo "  model: $model_override (dropped into Contents/Resources/tcn.json)"
    else
        echo "  model: baked-in default (plugins/$crate/assets/tcn.json)"
    fi
}

for p in "${PLUGINS[@]}"; do
    install_plugin "$p"
    echo
done

echo "Done. Restart your DAW (or rescan plugins) to pick up the build."

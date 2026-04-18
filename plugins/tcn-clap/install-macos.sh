#!/usr/bin/env bash
# Build the plugin, copy the CLAP + VST3 bundles into the user plugin dirs,
# optionally refresh the embedded tcn.json under Contents/Resources/, and
# re-codesign each bundle (required whenever bundle contents change).
#
# Usage:
#   ./install-macos.sh                  # uses the assets/tcn.json baked into the binary
#   ./install-macos.sh /path/to/tcn.json   # override with a specific model
#   TCN_CLAP_MODEL=/path ./install-macos.sh   # same, via env var

set -euo pipefail

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "install-macos.sh: macOS only (got OSTYPE=$OSTYPE)" >&2
    exit 1
fi

cd "$(dirname "$0")"

MODEL_OVERRIDE="${1:-${TCN_CLAP_MODEL:-}}"
BUILD_PROFILE="--release"

echo "==> Building tcn_clap ($BUILD_PROFILE)"
cargo xtask bundle tcn_clap $BUILD_PROFILE

BUNDLED="target/bundled"
CLAP_SRC="$BUNDLED/tcn_clap.clap"
VST3_SRC="$BUNDLED/tcn_clap.vst3"
CLAP_DEST="$HOME/Library/Audio/Plug-Ins/CLAP"
VST3_DEST="$HOME/Library/Audio/Plug-Ins/VST3"

install_bundle () {
    local kind="$1"                              # clap or vst3
    local src="$2"                               # source bundle path
    local dest_dir="$3"                          # destination plugin directory

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

    if [ -n "$MODEL_OVERRIDE" ]; then
        if [ ! -f "$MODEL_OVERRIDE" ]; then
            echo "TCN model override not found: $MODEL_OVERRIDE" >&2
            exit 1
        fi
        mkdir -p "$dest/Contents/Resources"
        cp "$MODEL_OVERRIDE" "$dest/Contents/Resources/tcn.json"
    fi

    # Re-codesign (ad-hoc) — mandatory whenever bundle contents change, even
    # the first time if we dropped tcn.json into Resources/. Without this
    # macOS treats the bundle as tampered and silently refuses to load it.
    codesign -s - --force --deep "$dest"
    echo "installed $dest"
}

echo "==> Installing bundles"
install_bundle clap "$CLAP_SRC" "$CLAP_DEST"
install_bundle vst3 "$VST3_SRC" "$VST3_DEST"

echo
if [ -n "$MODEL_OVERRIDE" ]; then
    echo "Model: $MODEL_OVERRIDE (dropped into Contents/Resources/tcn.json)"
else
    echo "Model: baked-in default (plugins/tcn-clap/assets/tcn.json)"
fi
echo "Restart your DAW (or rescan plugins) to pick up the build."

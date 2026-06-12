#!/usr/bin/env bash
#
# One-time setup: download the Kokoro-82M PyTorch model + voices from HuggingFace
# and convert them to the ONNX assets the `storytime` binary needs (assets/).
#
# Works on a bare Linux (Debian/Ubuntu, via apt) or macOS (via Homebrew) machine
# with nothing preinstalled but the system package manager. Safe to re-run.
#
# Usage:  ./setup.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

OS="$(uname -s)"

# ---- 1. system prerequisites (the only thing we shell out to apt/brew for) ----

install_linux_prereqs() {
    command -v apt-get >/dev/null 2>&1 || die \
        "this script's Linux path needs apt (Debian/Ubuntu). On another distro,
   install python3, python3-venv, python3-pip and espeak-ng with your package
   manager, then run:  python3 export/export.py"
    local SUDO=""
    [ "$(id -u)" -eq 0 ] || SUDO="sudo"
    log "Installing system packages via apt (python3, venv, pip, espeak-ng)"
    $SUDO apt-get update -y
    $SUDO apt-get install -y python3 python3-venv python3-pip espeak-ng
}

install_macos_prereqs() {
    command -v brew >/dev/null 2>&1 || die \
        "Homebrew not found. Install it from https://brew.sh and re-run."
    # Only install what's missing so re-runs are quick and quiet.
    command -v python3   >/dev/null 2>&1 || { log "brew install python";    brew install python; }
    command -v espeak-ng >/dev/null 2>&1 || { log "brew install espeak-ng"; brew install espeak-ng; }
}

case "$OS" in
    Linux)  install_linux_prereqs ;;
    Darwin) install_macos_prereqs ;;
    *)      die "unsupported OS: $OS (this script supports Linux and macOS)" ;;
esac

command -v python3 >/dev/null 2>&1 || die "python3 still not on PATH after install"

# ---- 2. isolated Python environment for the (build-time-only) export deps ----

VENV="$ROOT/export/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    log "Creating virtualenv at export/.venv"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

log "Upgrading pip"
python -m pip install --quiet --upgrade pip wheel setuptools

# CPU-only PyTorch on Linux: the default wheel bundles CUDA (~2 GB) which the
# export doesn't use. macOS wheels are already CPU/MPS-only.
if [ "$OS" = "Linux" ]; then
    log "Installing CPU PyTorch"
    python -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
fi

log "Installing export dependencies (export/requirements.txt)"
python -m pip install -r "$ROOT/export/requirements.txt"

# ---- 3. download from HuggingFace + convert to ONNX assets ----

log "Downloading Kokoro-82M from HuggingFace and exporting ONNX assets"
python "$ROOT/export/export.py"

# ---- 4. verify ----

ASSETS="$ROOT/assets"
[ -f "$ASSETS/kokoro.onnx" ] || die "export finished but $ASSETS/kokoro.onnx is missing"
[ -f "$ASSETS/tokens.json" ] || die "export finished but $ASSETS/tokens.json is missing"
voices=$(find "$ASSETS/voices" -name '*.bin' 2>/dev/null | wc -l | tr -d ' ')
[ "$voices" -gt 0 ] || die "export finished but no voices were written"
[ -f "$ASSETS/spk_encoder.onnx" ] || warn "spk_encoder.onnx missing (needed only for 'storytime clone')"

log "Success — assets ready in $ASSETS:"
printf '    kokoro.onnx  %s\n' "$(du -h "$ASSETS/kokoro.onnx" | cut -f1)"
printf '    tokens.json\n'
printf '    voices/      %s voice(s)\n' "$voices"
[ -f "$ASSETS/spk_encoder.onnx" ] && printf '    spk_encoder.onnx  %s (voice cloning)\n' "$(du -h "$ASSETS/spk_encoder.onnx" | cut -f1)"
echo
log "Next: build the CLI (needs Rust — https://rustup.rs):"
echo "    cd cli && cargo build --release            # ONNX backend"
echo "    cd cli && cargo build --release --features mlx   # + native MLX backend (Apple Silicon)"
echo
log "Then try it:"
echo "    echo 'həlˈoʊ.' | cli/target/release/storytime --ipa -o /tmp/hello.wav"

#!/usr/bin/env python3
"""Export Kokoro-82M to ONNX + raw voice tensors + token vocab.

Inputs come from a HuggingFace snapshot: either a local directory passed via
--snapshot, or (the default) downloaded from the Hub on demand. Outputs land in
../assets/ for the Rust CLI to consume.

Pipeline:
  kokoro-v1_0.pth  ->  kokoro.onnx
  voices/*.pt      ->  voices/*.bin   (float32, shape [511, 1, 256], C-order)
  config.json vocab ->  tokens.json   (char -> token-id map the CLI tokenizes with)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Stable upstream source. The revision is pinned to a known-good commit so a
# fresh setup is reproducible; pass --hf-revision main for the latest.
HF_REPO = "hexgrad/Kokoro-82M"
HF_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"


def download_snapshot(repo: str, revision: str) -> Path:
    """Fetch the weights, voices and config from the HuggingFace Hub.

    Only the files we need are pulled, into the standard HF cache; the local
    snapshot path is returned. Re-runs reuse the cache and work offline.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required to download the model.\n"
            "Run the repo's ./setup.sh, or: pip install huggingface_hub"
        )
    print(f"downloading {repo}@{revision[:12]} from HuggingFace ...", file=sys.stderr)
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=["config.json", "kokoro-v1_0.pth", "voices/*.pt"],
    )
    print(f"  snapshot at {path}", file=sys.stderr)
    return Path(path)


class KokoroONNXWrapper(nn.Module):
    """Wrap KModel so ONNX sees a clean (input_ids, style, speed) -> audio graph."""

    def __init__(self, kmodel):
        super().__init__()
        self.kmodel = kmodel

    def forward(self, input_ids: torch.Tensor, style: torch.Tensor, speed: torch.Tensor):
        # KModel.forward_with_tokens expects input_ids already padded with 0 on each side
        # and a style vector of shape [1, 256]. Returns (audio[T], pred_dur).
        audio, _ = self.kmodel.forward_with_tokens(input_ids, style, speed)
        return audio


def export_model(snapshot: Path, out_dir: Path, opset: int) -> Path:
    from kokoro import KModel

    config_path = snapshot / "config.json"
    weights_path = snapshot / "kokoro-v1_0.pth"
    if not weights_path.exists():
        sys.exit(f"model weights not found: {weights_path}")

    print(f"loading KModel from {weights_path.name} ...", file=sys.stderr)
    kmodel = KModel(config=str(config_path), model=str(weights_path), disable_complex=True)
    kmodel.eval()

    wrapper = KokoroONNXWrapper(kmodel).eval()

    # Dummy inputs matching the real call shape.
    # input_ids: [1, T] int64 with 0 pad on each side; T>=3 for dynamic export.
    dummy_ids = torch.zeros((1, 10), dtype=torch.int64)
    dummy_ids[0, 1:-1] = torch.tensor([50, 83, 54, 57, 63, 65, 68, 56], dtype=torch.int64)
    dummy_style = torch.zeros((1, 256), dtype=torch.float32)
    dummy_speed = torch.ones((1,), dtype=torch.float32)

    onnx_path = out_dir / "kokoro.onnx"
    print(f"exporting to {onnx_path} (opset {opset}) ...", file=sys.stderr)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_style, dummy_speed),
            str(onnx_path),
            input_names=["input_ids", "style", "speed"],
            output_names=["audio"],
            dynamic_axes={
                "input_ids": {1: "tokens"},
                "audio": {0: "samples"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # Sanity: reload with onnx to catch malformed exports early.
    import onnx
    onnx.checker.check_model(str(onnx_path))
    print(f"  ok: {onnx_path.stat().st_size / 1e6:.1f} MB", file=sys.stderr)
    return onnx_path


def export_voices(snapshot: Path, out_dir: Path) -> int:
    voices_in = snapshot / "voices"
    voices_out = out_dir / "voices"
    voices_out.mkdir(parents=True, exist_ok=True)
    count = 0
    for pt in sorted(voices_in.glob("*.pt")):
        t = torch.load(pt, map_location="cpu", weights_only=True)
        arr = t.detach().cpu().numpy().astype(np.float32, copy=False)
        # Expected shape [511, 1, 256]. Store contiguous little-endian float32.
        if arr.shape != (511, 1, 256):
            print(f"  warn: {pt.name} has shape {arr.shape}", file=sys.stderr)
        arr = np.ascontiguousarray(arr)
        (voices_out / f"{pt.stem}.bin").write_bytes(arr.tobytes(order="C"))
        count += 1
    print(f"wrote {count} voices to {voices_out}", file=sys.stderr)
    return count


def export_tokens(snapshot: Path, out_dir: Path) -> None:
    cfg = json.loads((snapshot / "config.json").read_text())
    vocab = cfg["vocab"]
    (out_dir / "tokens.json").write_text(
        json.dumps({"vocab": vocab, "n_token": cfg["n_token"]}, ensure_ascii=False, indent=2)
    )
    print(f"wrote tokens.json ({len(vocab)} entries)", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Kokoro-82M to ONNX assets")
    ap.add_argument(
        "--snapshot", type=Path, default=None,
        help="local HuggingFace snapshot dir; if omitted, download from the Hub",
    )
    ap.add_argument("--hf-repo", default=HF_REPO, help="HuggingFace repo to download")
    ap.add_argument(
        "--hf-revision", default=HF_REVISION,
        help="pinned commit/branch (default is a known-good commit; use 'main' for latest)",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "assets",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--skip-voices", action="store_true")
    args = ap.parse_args()

    if args.snapshot is None:
        snapshot = download_snapshot(args.hf_repo, args.hf_revision)
    elif args.snapshot.exists():
        snapshot = args.snapshot
    else:
        sys.exit(f"snapshot not found: {args.snapshot}")

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_model:
        export_model(snapshot, args.out, args.opset)
    export_tokens(snapshot, args.out)
    if not args.skip_voices:
        export_voices(snapshot, args.out)
    print("done.", file=sys.stderr)


if __name__ == "__main__":
    main()

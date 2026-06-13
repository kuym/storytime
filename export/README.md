# export/ — one-time model export

This directory holds the **build-time-only** Python step that turns the upstream
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) PyTorch release into the
assets the `storytime` Rust binary loads at runtime. Python is **not** a runtime
dependency — it runs once, offline, to produce `../assets/`.

`export.py` downloads the pinned Kokoro checkpoint, voices, and config from
HuggingFace and writes:

- `../assets/kokoro.onnx` — the model as a clean `(input_ids, style, speed) → audio` ONNX graph
- `../assets/tokens.json` — the IPA-char → token-id vocabulary
- `../assets/voices/*.bin` — the voicepacks as raw little-endian f32 `[N, 1, 256]`
- `../assets/spk_encoder.onnx` — the GE2E speaker encoder used by `storytime clone` (voice cloning)

## Run it

The repo's top-level `./setup.sh` does all of this for you (system prereqs, a
venv, then this script). To run it directly into an existing Python env:

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python export.py                 # downloads from HuggingFace, writes ../assets/
```

Useful flags: `--snapshot DIR` (use a local HuggingFace snapshot instead of
downloading), `--skip-model` / `--skip-voices` / `--skip-spk` (skip individual
outputs), `--hf-revision main` (use latest instead of the pinned commit).
`verify_spk.py` checks the exported speaker encoder against Resemblyzer; see the
header in that file.

See the main [README.md](../README.md) (the **Setup** and **Voice cloning**
sections) for full details, prerequisites, and how the assets are consumed.

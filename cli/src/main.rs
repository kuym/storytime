// storytime: Kokoro-82M TTS CLI.
//
// Reads text or IPA phonemes from stdin, runs Kokoro via ONNX Runtime
// (CoreML EP on Apple Silicon, CPU fallback), and writes a WAV file.
//
// Text mode shells out to `espeak-ng` for grapheme->IPA conversion.
// IPA mode (--ipa) skips that step so the tool composes in a POSIX pipeline.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use hound::{SampleFormat, WavSpec, WavWriter};
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

// ort::Error doesn't implement std::error::Error; bridge via Display.
fn ort_err<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("ort: {e}")
}
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use serde::Deserialize;

const NATIVE_SR: u32 = 24_000;
const STYLE_DIM: usize = 256;
// Most voices are 511 rows; some are 510. Loader accepts either.
#[allow(dead_code)]
const MAX_VOICE_ROWS: usize = 511;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BitDepth {
    #[value(name = "16")]
    I16,
    #[value(name = "24")]
    I24,
    #[value(name = "32")]
    I32,
    #[value(name = "float32")]
    F32,
}

#[derive(Parser, Debug)]
#[command(
    name = "storytime",
    about = "Kokoro-82M text-to-speech (ONNX Runtime + CoreML).",
    long_about = "Reads UTF-8 from stdin. Default mode expects raw text and \
                  pipes through `espeak-ng` for IPA phonemization. With --ipa, \
                  stdin is treated as IPA phonemes directly (as espeak-ng -q --ipa=3 \
                  would emit), and espeak-ng is not required."
)]
struct Args {
    /// Voice name (see --list-voices), e.g. af_bella.
    #[arg(long, default_value = "af_heart")]
    voice: String,

    /// Output WAV sample rate in Hz. Model is 24000 Hz; other rates are resampled.
    #[arg(long, default_value_t = 24_000)]
    sample_rate: u32,

    /// Output PCM bit depth.
    #[arg(long, value_enum, default_value_t = BitDepth::I16)]
    bit_depth: BitDepth,

    /// Speaking rate multiplier.
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Treat stdin as IPA phonemes rather than text (no espeak-ng invocation).
    #[arg(long)]
    ipa: bool,

    /// Directory holding kokoro.onnx, tokens.json, voices/*.bin.
    /// Defaults to ../assets relative to the binary.
    #[arg(long)]
    assets: Option<PathBuf>,

    /// List available voices and exit.
    #[arg(long)]
    list_voices: bool,

    /// Output WAV path.
    #[arg(short = 'o', long, default_value = "out.wav")]
    output: PathBuf,
}

#[derive(Deserialize)]
struct TokensFile {
    vocab: HashMap<String, i64>,
    #[allow(dead_code)]
    n_token: usize,
}

fn assets_dir(cli_override: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = cli_override {
        return Ok(p.to_path_buf());
    }
    // Look relative to the executable, then relative to CWD.
    if let Ok(exe) = std::env::current_exe() {
        for up in [1, 2, 3] {
            if let Some(mut d) = exe.ancestors().nth(up).map(Path::to_path_buf) {
                d.push("assets");
                if d.join("kokoro.onnx").exists() {
                    return Ok(d);
                }
            }
        }
    }
    let cwd = std::env::current_dir()?.join("assets");
    if cwd.join("kokoro.onnx").exists() {
        return Ok(cwd);
    }
    bail!("could not locate assets/ directory (pass --assets)")
}

fn list_voices(assets: &Path) -> Result<()> {
    let dir = assets.join("voices");
    let mut names: Vec<_> = fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("bin") {
                p.file_stem().and_then(|s| s.to_str()).map(String::from)
            } else {
                None
            }
        })
        .collect();
    names.sort();
    for n in names {
        println!("{n}");
    }
    Ok(())
}

fn load_tokens(assets: &Path) -> Result<HashMap<String, i64>> {
    let raw = fs::read_to_string(assets.join("tokens.json"))?;
    let t: TokensFile = serde_json::from_str(&raw)?;
    Ok(t.vocab)
}

/// Tokenize IPA input by grapheme-cluster lookup against the vocab.
/// Unknown characters are silently dropped (matches upstream behavior).
fn tokenize(ipa: &str, vocab: &HashMap<String, i64>) -> Vec<i64> {
    let mut out = Vec::with_capacity(ipa.len());
    for ch in ipa.chars() {
        // vocab keys are single chars in config.json.
        let mut buf = [0u8; 4];
        let key = ch.encode_utf8(&mut buf);
        if let Some(&id) = vocab.get(key) {
            out.push(id);
        }
    }
    out
}

fn run_espeak(text: &str) -> Result<String> {
    let mut child = Command::new("espeak-ng")
        .args(["-q", "--ipa=3", "-v", "en-us"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn espeak-ng (install it or use --ipa)")?;
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(text.as_bytes())?;
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!("espeak-ng failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    Ok(String::from_utf8(out.stdout)?
        .chars()
        .filter(|c| !c.is_control() || *c == '\n')
        .collect::<String>()
        .replace('\n', " ")
        .trim()
        .to_string())
}

struct Voice {
    data: Vec<f32>,
    rows: usize,
}

fn load_voice(assets: &Path, name: &str) -> Result<Voice> {
    let path = assets.join("voices").join(format!("{name}.bin"));
    let bytes = fs::read(&path).with_context(|| format!("voice not found: {}", path.display()))?;
    if bytes.len() % (STYLE_DIM * 4) != 0 {
        bail!("voice {name}: unexpected size {} bytes", bytes.len());
    }
    let rows = bytes.len() / (STYLE_DIM * 4);
    let mut data = Vec::with_capacity(rows * STYLE_DIM);
    for chunk in bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Voice { data, rows })
}

/// Select the style row for a given token length. Reference runtime indexes
/// by the number of phoneme tokens (unpadded), clamped to the voice length.
fn select_style(voice: &Voice, n_tokens: usize) -> Vec<f32> {
    let idx = n_tokens.min(voice.rows - 1);
    let start = idx * STYLE_DIM;
    voice.data[start..start + STYLE_DIM].to_vec()
}

fn resample(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(samples.to_vec());
    }
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut r = SincFixedIn::<f32>::new(
        to as f64 / from as f64,
        2.0,
        params,
        samples.len(),
        1,
    )?;
    let out = r.process(&[samples.to_vec()], None)?;
    Ok(out.into_iter().next().unwrap())
}

fn write_wav(path: &Path, samples: &[f32], sr: u32, depth: BitDepth) -> Result<()> {
    let (bits, fmt) = match depth {
        BitDepth::I16 => (16u16, SampleFormat::Int),
        BitDepth::I24 => (24, SampleFormat::Int),
        BitDepth::I32 => (32, SampleFormat::Int),
        BitDepth::F32 => (32, SampleFormat::Float),
    };
    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: bits,
        sample_format: fmt,
    };
    let mut w = WavWriter::create(path, spec)?;
    match depth {
        BitDepth::F32 => {
            for &s in samples {
                w.write_sample(s.clamp(-1.0, 1.0))?;
            }
        }
        BitDepth::I16 => {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                w.write_sample(v)?;
            }
        }
        BitDepth::I24 => {
            let peak = ((1i32 << 23) - 1) as f32;
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) * peak) as i32;
                w.write_sample(v)?;
            }
        }
        BitDepth::I32 => {
            for &s in samples {
                let v = (s.clamp(-1.0, 1.0) as f64 * i32::MAX as f64) as i32;
                w.write_sample(v)?;
            }
        }
    }
    w.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let assets = assets_dir(args.assets.as_deref())?;

    if args.list_voices {
        return list_voices(&assets);
    }

    let mut stdin_buf = String::new();
    std::io::stdin().read_to_string(&mut stdin_buf)?;
    let input = stdin_buf.trim();
    if input.is_empty() {
        bail!("stdin was empty");
    }

    let ipa = if args.ipa {
        input.to_string()
    } else {
        run_espeak(input)?
    };

    let vocab = load_tokens(&assets)?;
    let tokens = tokenize(&ipa, &vocab);
    if tokens.is_empty() {
        bail!("no valid phoneme tokens produced from input");
    }
    let voice = load_voice(&assets, &args.voice)?;
    if tokens.len() > voice.rows - 1 {
        bail!(
            "input too long: {} tokens (voice {} supports up to {})",
            tokens.len(),
            args.voice,
            voice.rows - 1
        );
    }
    let style = select_style(&voice, tokens.len());

    // Pad with 0 on both sides per model convention.
    let mut padded = Vec::with_capacity(tokens.len() + 2);
    padded.push(0);
    padded.extend_from_slice(&tokens);
    padded.push(0);

    eprintln!(
        "storytime: {} phonemes, voice={}, speed={}",
        tokens.len(),
        args.voice,
        args.speed
    );

    // Build ONNX session with CoreML EP (falls back to CPU automatically).
    ort::init().with_name("storytime").commit().map_err(ort_err)?;
    let mut session = Session::builder()
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .map_err(ort_err)?
        .commit_from_file(assets.join("kokoro.onnx"))
        .map_err(ort_err)?;

    let ids_shape = vec![1_i64, padded.len() as i64];
    let style_shape = vec![1_i64, STYLE_DIM as i64];
    let speed_shape = vec![1_i64];

    let ids_t = Tensor::from_array((ids_shape, padded)).map_err(ort_err)?;
    let style_t = Tensor::from_array((style_shape, style)).map_err(ort_err)?;
    let speed_t = Tensor::from_array((speed_shape, vec![args.speed])).map_err(ort_err)?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => ids_t,
            "style" => style_t,
            "speed" => speed_t,
        ])
        .map_err(ort_err)?;

    let (_shape, audio) = outputs["audio"]
        .try_extract_tensor::<f32>()
        .map_err(ort_err)?;
    let samples: Vec<f32> = audio.to_vec();
    eprintln!(
        "storytime: {} samples @ {} Hz ({:.2}s)",
        samples.len(),
        NATIVE_SR,
        samples.len() as f32 / NATIVE_SR as f32
    );

    let resampled = resample(&samples, NATIVE_SR, args.sample_rate)?;
    write_wav(&args.output, &resampled, args.sample_rate, args.bit_depth)?;
    eprintln!("storytime: wrote {}", args.output.display());
    Ok(())
}

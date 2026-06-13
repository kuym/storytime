//! `storytime clone`: create a new voicepack from a short reference recording.
//!
//! Kokoro ships no reference/style encoder, so cloning is a gradient-free
//! search (after KVoiceWalk, Apache-2.0): start from a blend of stock
//! voicepacks, perturb a single 256-d style delta shared across all rows,
//! synthesize fixed test utterances with the normal backend, and score the
//! audio against the reference recording. The result is written as an
//! ordinary `voices/<name>.bin`, so after this one-time step `--voice <name>`
//! works everywhere with zero runtime cost. See docs/voice-cloning.md.
//!
//! Deliberate deviations from upstream KVoiceWalk: the walk moves one shared
//! per-dimension delta instead of the whole [510,1,256] tensor (upstream only
//! ever *evaluates* the one or two rows its test texts select, so the other
//! ~508 rows drift as unguided noise), feature differences are NaN-safe, and
//! the RNG is seedable.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::dsp::{self, AcousticStats, Rng, SpeakerEncoder, SPK_SR};
use crate::{Backend, Runtime, Voice, NATIVE_SR, STYLE_DIM};

/// Weighted harmonic mean (KVoiceWalk's validated formula): speaker
/// similarity to the reference, cross-text self-similarity (stability),
/// acoustic-feature guard rail.
const W_TARGET: f32 = 0.48;
const W_SELF: f32 = 0.50;
const W_FEATURE: f32 = 0.02;
/// Early-exit gate: skip the second synthesis when the candidate's target
/// similarity drops below this fraction of the current best's.
const GATE: f32 = 0.98;
/// Per-step perturbation scale is the stock voices' per-dim std times a
/// fresh uniform draw from this range (upstream's "diversity").
const DIVERSITY: (f32, f32) = (0.01, 0.15);
/// Reject candidates whose target-utterance duration strays this far from
/// the init blend's (degenerate styles produce rushed or droning audio).
const MAX_DURATION_DRIFT: f32 = 0.30;
/// Trim threshold applied to reference and candidate audio alike.
const TRIM: f32 = 0.005;
/// How many stock voices the automatic init blends.
const AUTO_INIT_TOP: usize = 3;
/// Checkpoint cadence, in acceptances.
const CHECKPOINT_EVERY: u32 = 50;

/// The paragraph the user records themselves reading (`--print-script`).
/// Its IPA is baked below (`TARGET_IPA`) so espeak-ng is not needed at clone
/// time and the speaker-embedding comparison is text-matched.
pub const REFERENCE_SCRIPT: &str = "\
The quick autumn rain washed over the village, and every rooftop shone like \
polished silver. Father Bear measured out three bowls of porridge, humming an \
old tune, while the youngest child counted bright stars through the window.";

/// Second fixed utterance, synthesized only to measure the candidate's
/// cross-text stability (self-similarity). Never recorded by the user.
/// Only read at IPA-regeneration time (see `regenerate_baked_ipa`).
#[allow(dead_code)]
const OTHER_TEXT: &str = "\
Long before sunrise, the baker stirred warm milk into golden dough, \
whispering numbers while the ovens ticked and glowed. Outside, a patient \
grey donkey watched the first snowflakes settle on the quiet market square.";

// Baked espeak-ng output for the two texts above. Regenerate after editing
// either text:  cargo test regenerate_baked_ipa -- --ignored --nocapture
const TARGET_IPA: &str = "ðə kwˈɪk ˈɔːɾʌm ɹˈe\u{200d}ɪn wˈɑːʃt ˌo\u{200d}ʊvɚ ðə vˈɪlɪd\u{200d}ʒ, ænd ˈɛvɹi ɹˈuːftɑːp ʃˈɑːn lˈa\u{200d}ɪk pˈɑːlɪʃt sˈɪlvɚ. fˈɑːðɚ bˈɛ\u{200d}ɹ mˈɛʒɚd ˈa\u{200d}ʊt θɹˈiː bˈo\u{200d}ʊlz ʌv pˈɔɹɪd\u{200d}ʒ, hˈʌmɪŋ ɐn ˈo\u{200d}ʊld tˈuːn, wˌa\u{200d}ɪl ðə jˈʌŋɡɪst t\u{200d}ʃˈa\u{200d}ɪld kˈa\u{200d}ʊntᵻd bɹˈa\u{200d}ɪt stˈɑː\u{200d}ɹz θɹuː ðə wˈɪndo\u{200d}ʊ.";
const OTHER_IPA: &str = "lˈɔŋ bᵻfˌɔː\u{200d}ɹ sˈʌnɹa\u{200d}ɪz, ðə bˈe\u{200d}ɪkɚ stˈɜːd wˈɔː\u{200d}ɹm mˈɪlk ˌɪntʊ ɡˈo\u{200d}ʊldən dˈo\u{200d}ʊ, wˈɪspɚɹɪŋ nˈʌmbɚz wˌa\u{200d}ɪl ðɪ ˈʌvənz tˈɪkt ænd ɡlˈo\u{200d}ʊd. a\u{200d}ʊtsˈa\u{200d}ɪd, ɐ pˈe\u{200d}ɪʃənt ɡɹˈe\u{200d}ɪ dˈɔŋki wˈɑːt\u{200d}ʃt ðə fˈɜːst snˈo\u{200d}ʊfle\u{200d}ɪks sˈɛɾə\u{200d}l ɔnðə kwˈa\u{200d}ɪ\u{200d}ət mˈɑː\u{200d}ɹkɪt skwˈɛ\u{200d}ɹ.";

#[derive(clap::Args, Debug)]
pub struct CloneArgs {
    /// Reference recording of the target speaker reading the reference script
    /// (10-20 s WAV, mono or stereo; see --print-script).
    #[arg(long = "ref", required_unless_present = "print_script")]
    pub reference: Option<PathBuf>,

    /// Name of the new voice; written to <assets>/voices/<name>.bin.
    #[arg(long, required_unless_present = "print_script")]
    pub name: Option<String>,

    /// Print the reference script to read aloud when recording, then exit.
    #[arg(long)]
    pub print_script: bool,

    /// Voicepacks to blend as the starting point (comma-separated). Default:
    /// rank all American/British voices against the reference and blend the
    /// closest three.
    #[arg(long, value_delimiter = ',')]
    pub init: Vec<String>,

    /// Maximum optimization steps.
    #[arg(long, default_value_t = 2000)]
    pub steps: u32,

    /// Wall-clock budget in minutes; stops at whichever of --steps or this
    /// limit hits first (0 = no time limit).
    #[arg(long, default_value_t = 0)]
    pub budget_min: u32,

    /// RNG seed. Reproducibility is best-effort: CoreML/Metal inference is
    /// not bit-deterministic, so accepted paths can diverge between runs.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Resume an interrupted walk from <name>'s checkpoint.
    #[arg(long)]
    pub resume: bool,

    /// Transcript of the reference recording, if it is not the built-in
    /// reference script. Requires espeak-ng on PATH.
    #[arg(long)]
    pub ref_text: Option<PathBuf>,

    /// Directory holding kokoro.onnx, tokens.json, voices/*.bin,
    /// spk_encoder.onnx. Defaults to ../assets relative to the binary.
    #[arg(long)]
    pub assets: Option<PathBuf>,

    /// Inference backend for the synthesis loop (see the top-level --backend).
    #[arg(long, value_enum)]
    pub backend: Option<Backend>,

    /// Directory to cache the CoreML-compiled model between runs.
    #[arg(long)]
    pub coreml_cache: Option<PathBuf>,

    /// Disable the CoreML compiled-model cache.
    #[arg(long)]
    pub no_coreml_cache: bool,
}

// ---------------------------------------------------------------------------
// Voicepack math
// ---------------------------------------------------------------------------

/// Per-dimension statistics across the installed stock voicepacks (all rows
/// of all voices): the std scales perturbations, min/max clamp the walk to
/// the trained style manifold.
struct Envelope {
    std: Vec<f32>,
    min: Vec<f32>,
    max: Vec<f32>,
}

fn envelope<'a>(voices: impl IntoIterator<Item = &'a Voice>) -> Envelope {
    let mut n = 0u64;
    let mut sum = vec![0f64; STYLE_DIM];
    let mut sum_sq = vec![0f64; STYLE_DIM];
    let mut min = vec![f32::INFINITY; STYLE_DIM];
    let mut max = vec![f32::NEG_INFINITY; STYLE_DIM];
    for v in voices {
        for row in v.data.chunks_exact(STYLE_DIM) {
            n += 1;
            for (c, &x) in row.iter().enumerate() {
                sum[c] += x as f64;
                sum_sq[c] += (x as f64) * (x as f64);
                min[c] = min[c].min(x);
                max[c] = max[c].max(x);
            }
        }
    }
    let std = (0..STYLE_DIM)
        .map(|c| {
            let mean = sum[c] / n as f64;
            ((sum_sq[c] / n as f64 - mean * mean).max(0.0)).sqrt() as f32
        })
        .collect();
    Envelope { std, min, max }
}

/// Weighted mean of voicepacks (weights need not be normalized). Voices with
/// 511 rows blend with 510-row ones by truncating to the shortest.
fn blend_voices(voices: &[(Voice, f32)]) -> Voice {
    let rows = voices.iter().map(|(v, _)| v.rows).min().expect("non-empty blend");
    let total_w: f32 = voices.iter().map(|(_, w)| w).sum();
    let mut data = vec![0f32; rows * STYLE_DIM];
    for (v, w) in voices {
        let w = w / total_w;
        for (acc, &x) in data.iter_mut().zip(&v.data) {
            *acc += w * x;
        }
    }
    Voice { data, rows }
}

/// The style fed to the model: the init blend's row for this token count,
/// shifted by the walk's delta and clamped to the stock-voice envelope.
fn style_for(base: &Voice, n_tokens: usize, delta: &[f32], env: &Envelope) -> Vec<f32> {
    let mut style = crate::select_style(base, n_tokens);
    for (c, s) in style.iter_mut().enumerate() {
        *s = (*s + delta[c]).clamp(env.min[c], env.max[c]);
    }
    style
}

/// Serialize `base + delta` (clamped) as a voicepack in the exact format
/// `load_voice` reads: rows x 256 contiguous little-endian f32.
fn write_voice_bin(path: &Path, base: &Voice, delta: &[f32], env: &Envelope) -> Result<()> {
    let mut bytes = Vec::with_capacity(base.rows * STYLE_DIM * 4);
    for row in base.data.chunks_exact(STYLE_DIM) {
        for (c, &x) in row.iter().enumerate() {
            let v = (x + delta[c]).clamp(env.min[c], env.max[c]);
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct Score {
    total: f32,
    target_sim: f32,
    self_sim: f32,
    feat_sim: f32,
}

fn harmonic_score(target_sim: f32, self_sim: f32, feat_sim: f32) -> f32 {
    let weights = [W_TARGET, W_SELF, W_FEATURE];
    let values = [target_sim.max(1e-6), self_sim.max(1e-6), feat_sim.max(1e-6)];
    let wsum: f32 = weights.iter().sum();
    let denom: f32 = weights.iter().zip(values).map(|(w, v)| w / v).sum();
    wsum / denom * 100.0
}

/// Everything the per-candidate evaluation needs.
struct EvalCtx<'a> {
    rt: &'a mut Runtime,
    enc: &'a mut SpeakerEncoder,
    base: &'a Voice,
    env: &'a Envelope,
    target_tokens: &'a [i64],
    other_tokens: &'a [i64],
    ref_emb: [f32; dsp::EMBED_DIM],
    ref_stats: AcousticStats,
    /// Trimmed sample count of the init blend's target synthesis; the
    /// duration sanity guard measures drift against this (0 = guard off).
    base_target_len: usize,
}

impl EvalCtx<'_> {
    /// Synthesize one utterance with the candidate style and return trimmed
    /// 16 kHz audio.
    fn synth_16k(&mut self, tokens: &[i64], delta: &[f32]) -> Result<Vec<f32>> {
        let style = style_for(self.base, tokens.len(), delta, self.env);
        let audio = self.rt.synth(tokens, style, 1.0)?;
        let trimmed = crate::trim_silence(&audio, TRIM);
        if trimmed.is_empty() {
            bail!("candidate produced silent audio");
        }
        crate::resample(trimmed, NATIVE_SR, SPK_SR)
    }

    /// Score a candidate delta. `gate_target_sim` short-circuits hopeless
    /// candidates after the first synthesis (pass 0.0 to disable). `None`
    /// means rejected by the gate or a sanity guard, not an error.
    fn evaluate(&mut self, delta: &[f32], gate_target_sim: f32) -> Result<Option<Score>> {
        let target_16k = match self.synth_16k(self.target_tokens, delta) {
            Ok(a) => a,
            Err(_) => return Ok(None), // degenerate style; treat as rejection
        };
        if self.base_target_len > 0 {
            let drift = (target_16k.len() as f32 - self.base_target_len as f32).abs()
                / self.base_target_len as f32;
            if drift > MAX_DURATION_DRIFT {
                return Ok(None);
            }
        }
        let target_emb = self.enc.embed(&target_16k)?;
        let target_sim = dsp::cosine(&target_emb, &self.ref_emb);
        if target_sim <= gate_target_sim {
            return Ok(None);
        }

        let other_16k = match self.synth_16k(self.other_tokens, delta) {
            Ok(a) => a,
            Err(_) => return Ok(None),
        };
        let other_emb = self.enc.embed(&other_16k)?;
        let self_sim = dsp::cosine(&target_emb, &other_emb);
        let feat_sim = dsp::feature_similarity(
            &dsp::acoustic_stats(&target_16k, SPK_SR),
            &self.ref_stats,
        );
        Ok(Some(Score {
            total: harmonic_score(target_sim, self_sim, feat_sim),
            target_sim,
            self_sim,
            feat_sim,
        }))
    }
}

// ---------------------------------------------------------------------------
// Checkpointing
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    version: u32,
    name: String,
    seed: u64,
    step: u32,
    accepted: u32,
    best_score: f32,
    best_target_sim: f32,
    /// Init blend as (voice name, weight) pairs, so --resume rebuilds the
    /// identical base without re-ranking.
    init: Vec<(String, f32)>,
    delta: Vec<f32>,
}

fn checkpoint_path(assets: &Path, name: &str) -> PathBuf {
    assets.join("voices").join(format!("{name}.clone.json"))
}

fn save_checkpoint(assets: &Path, ck: &Checkpoint, base: &Voice, env: &Envelope) -> Result<()> {
    let path = checkpoint_path(assets, &ck.name);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string(ck)?)?;
    std::fs::rename(&tmp, &path)?;
    // Also refresh the auditionable voicepack: `--voice <name>` mid-run.
    write_voice_bin(
        &assets.join("voices").join(format!("{}.bin", ck.name)),
        base,
        &ck.delta,
        env,
    )
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(args: CloneArgs) -> Result<()> {
    if args.print_script {
        println!("{REFERENCE_SCRIPT}");
        return Ok(());
    }
    let name = args.name.as_deref().expect("clap enforces --name");
    let ref_path = args.reference.as_deref().expect("clap enforces --ref");
    if name.contains(['/', '\\']) || name.is_empty() {
        bail!("--name must be a bare voice name (it becomes voices/<name>.bin)");
    }

    let assets = crate::assets_dir(args.assets.as_deref())?;
    let vocab = crate::load_tokens(&assets)?;

    // Resolve the backend up front: it selects not only Kokoro synthesis but
    // also the speaker encoder (MLX runs both on the GPU; ONNX uses ort-CPU for
    // the encoder), so it must be known before the reference is embedded.
    let backend = args.backend.unwrap_or(if cfg!(feature = "mlx") {
        Backend::Mlx
    } else {
        Backend::Onnx
    });

    // --- Reference recording -> embedding + acoustic stats.
    let (ref_wav, ref_sr) = dsp::read_wav_mono(ref_path)?;
    let secs = ref_wav.len() as f32 / ref_sr as f32;
    if !(4.0..=120.0).contains(&secs) {
        bail!(
            "reference is {secs:.1}s; record roughly 10-20s of clean speech \
             (storytime clone --print-script for the text to read)"
        );
    }
    let ref_16k_full = crate::resample(&ref_wav, ref_sr, SPK_SR)?;
    let ref_16k = crate::trim_silence(&ref_16k_full, TRIM).to_vec();
    if ref_16k.is_empty() {
        bail!("reference recording is silent");
    }
    let mut enc = SpeakerEncoder::load(&assets, backend)?;
    let ref_emb = enc.embed(&ref_16k)?;
    let ref_stats = dsp::acoustic_stats(&ref_16k, SPK_SR);
    eprintln!(
        "clone: reference {:.1}s @ {} Hz, f0 {:.0}±{:.0} Hz",
        secs, ref_sr, ref_stats.f0_mean, ref_stats.f0_std
    );

    // --- Test utterances -> token sequences (baked IPA by default).
    let target_ipa = match &args.ref_text {
        None => TARGET_IPA.to_string(),
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading --ref-text {}", path.display()))?;
            crate::run_espeak(&crate::normalize_punctuation(text.trim()))?
        }
    };
    let target_tokens = crate::tokenize(&target_ipa, &vocab);
    let other_tokens = crate::tokenize(OTHER_IPA, &vocab);
    if target_tokens.is_empty() || other_tokens.is_empty() {
        bail!("test utterances produced no tokens (corrupt tokens.json?)");
    }

    // --- Stock voicepacks: perturbation envelope + init blend.
    let stock_names = crate::voice_names(&assets)?;
    if stock_names.is_empty() {
        bail!("no voices found under {}/voices", assets.display());
    }
    let stock: Vec<(String, Voice)> = stock_names
        .iter()
        .filter(|n| !n.contains(".clone")) // never feed checkpoints back in
        .filter(|n| n.as_str() != name)
        .map(|n| crate::load_voice(&assets, n).map(|v| (n.clone(), v)))
        .collect::<Result<_>>()?;
    let env = envelope(stock.iter().map(|(_, v)| v));
    let max_tokens = stock.iter().map(|(_, v)| v.rows - 1).min().unwrap();
    if target_tokens.len() > max_tokens || other_tokens.len() > max_tokens {
        bail!(
            "test utterance too long ({} tokens, cap {max_tokens}); shorten --ref-text",
            target_tokens.len().max(other_tokens.len())
        );
    }

    let cache = crate::resolve_coreml_cache(args.no_coreml_cache, args.coreml_cache.as_deref())?;
    let mut rt = Runtime::init(backend, &assets, cache)?;

    // --- Resume or fresh init.
    let ck_path = checkpoint_path(&assets, name);
    let mut checkpoint: Option<Checkpoint> = None;
    if args.resume {
        let raw = std::fs::read_to_string(&ck_path)
            .with_context(|| format!("--resume: no checkpoint at {}", ck_path.display()))?;
        let ck: Checkpoint = serde_json::from_str(&raw)?;
        if ck.delta.len() != STYLE_DIM {
            bail!("checkpoint {} is corrupt (delta length)", ck_path.display());
        }
        if !args.init.is_empty() {
            bail!("--resume restores the original --init; don't pass both");
        }
        checkpoint = Some(ck);
    } else if ck_path.exists() {
        bail!(
            "{} exists from a previous run; pass --resume to continue it or \
             delete it (and voices/{name}.bin) to start over",
            ck_path.display()
        );
    }

    let init_weights: Vec<(String, f32)> = match &checkpoint {
        Some(ck) => ck.init.clone(),
        None if !args.init.is_empty() => {
            args.init.iter().map(|n| (n.clone(), 1.0)).collect()
        }
        None => rank_voices(&stock, &target_tokens, &ref_emb, &env, &mut rt, &mut enc)?,
    };
    eprintln!(
        "clone: init blend: {}",
        init_weights
            .iter()
            .map(|(n, w)| format!("{n}:{w:.2}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let base = {
        let mut parts = Vec::new();
        for (n, w) in &init_weights {
            let v = stock
                .iter()
                .find(|(sn, _)| sn == n)
                .map(|(_, v)| Voice { data: v.data.clone(), rows: v.rows })
                .ok_or_else(|| anyhow::anyhow!("init voice {n} not found in assets"))?;
            parts.push((v, *w));
        }
        blend_voices(&parts)
    };

    let mut ctx = EvalCtx {
        rt: &mut rt,
        enc: &mut enc,
        base: &base,
        env: &env,
        target_tokens: &target_tokens,
        other_tokens: &other_tokens,
        ref_emb,
        ref_stats,
        base_target_len: 0,
    };

    // Baseline: measure the init blend itself (delta = 0).
    let zero = vec![0f32; STYLE_DIM];
    let baseline_16k = ctx.synth_16k(&target_tokens, &zero)?;
    ctx.base_target_len = baseline_16k.len();
    let baseline = ctx
        .evaluate(&zero, 0.0)?
        .expect("baseline evaluation cannot be gated");

    let (mut best_delta, mut best, mut step, mut accepted) = match checkpoint {
        Some(ck) => {
            let delta = ck.delta.clone();
            let restored = ctx
                .evaluate(&delta, 0.0)?
                .ok_or_else(|| anyhow::anyhow!("checkpoint delta no longer evaluates"))?;
            eprintln!(
                "clone: resumed at step {} (score {:.2}, target sim {:.3})",
                ck.step, restored.total, restored.target_sim
            );
            (delta, restored, ck.step, ck.accepted)
        }
        None => {
            eprintln!(
                "clone: baseline score {:.2} (target sim {:.3}, self {:.3}, feat {:.3})",
                baseline.total, baseline.target_sim, baseline.self_sim, baseline.feat_sim
            );
            (zero.clone(), baseline, 0, 0)
        }
    };

    // --- The walk: greedy hill climb.
    let mut rng = Rng::new(args.seed.wrapping_add(step as u64));
    let started = Instant::now();
    let budget = (args.budget_min as u64).checked_mul(60).unwrap_or(0);
    let mut last_heartbeat = Instant::now();
    let mut candidate = vec![0f32; STYLE_DIM];
    while step < args.steps {
        if budget > 0 && started.elapsed().as_secs() >= budget {
            eprintln!("clone: budget reached after {step} steps");
            break;
        }
        step += 1;

        let diversity = rng.uniform_in(DIVERSITY.0, DIVERSITY.1);
        for (c, v) in candidate.iter_mut().enumerate() {
            *v = best_delta[c] + rng.gauss() * ctx.env.std[c] * diversity;
        }
        let gate = best.target_sim * GATE;
        if let Some(score) = ctx.evaluate(&candidate, gate)? {
            if score.total > best.total {
                accepted += 1;
                best = score;
                best_delta.copy_from_slice(&candidate);
                eprintln!(
                    "clone: step {step}/{} score {:.2} (target {:.3} self {:.3} feat {:.3}) \
                     diversity {:.3} [accept #{accepted}]",
                    args.steps, best.total, best.target_sim, best.self_sim, best.feat_sim,
                    diversity
                );
                if accepted % CHECKPOINT_EVERY == 0 {
                    let ck = Checkpoint {
                        version: 1,
                        name: name.to_string(),
                        seed: args.seed,
                        step,
                        accepted,
                        best_score: best.total,
                        best_target_sim: best.target_sim,
                        init: init_weights.clone(),
                        delta: best_delta.clone(),
                    };
                    save_checkpoint(&assets, &ck, ctx.base, ctx.env)?;
                    eprintln!("clone: checkpoint saved (audition with --voice {name})");
                }
            }
        }

        if last_heartbeat.elapsed().as_secs() >= 30 {
            let rate = step as f32 / started.elapsed().as_secs_f32();
            eprintln!(
                "clone: step {step}/{} best {:.2} target-sim {:.3} ({rate:.2} steps/s, {} accepted)",
                args.steps, best.total, best.target_sim, accepted
            );
            last_heartbeat = Instant::now();
        }
    }

    // --- Final artifacts.
    let ck = Checkpoint {
        version: 1,
        name: name.to_string(),
        seed: args.seed,
        step,
        accepted,
        best_score: best.total,
        best_target_sim: best.target_sim,
        init: init_weights.clone(),
        delta: best_delta.clone(),
    };
    save_checkpoint(&assets, &ck, ctx.base, ctx.env)?;
    eprintln!(
        "clone: done after {step} steps ({accepted} accepted) in {:.0}s",
        started.elapsed().as_secs_f32()
    );
    eprintln!(
        "clone: final score {:.2}: target sim {:.3} (baseline {:.3}), self {:.3}, feat {:.3}",
        best.total, best.target_sim, baseline.target_sim, best.self_sim, best.feat_sim
    );
    eprintln!("clone: wrote voices/{name}.bin — try: echo \"Once upon a time...\" | storytime --voice {name}");
    eprintln!("clone: re-run with --resume to keep optimizing, or a new --seed for a fresh walk");
    Ok(())
}

/// Rank stock voices by speaker similarity to the reference and return the
/// top blend, similarity-weighted. Only English (a*/b*) voices are ranked:
/// the GE2E scorer is English-trained.
fn rank_voices(
    stock: &[(String, Voice)],
    target_tokens: &[i64],
    ref_emb: &[f32; dsp::EMBED_DIM],
    env: &Envelope,
    rt: &mut Runtime,
    enc: &mut SpeakerEncoder,
) -> Result<Vec<(String, f32)>> {
    let zero = vec![0f32; STYLE_DIM];
    let mut ranked: Vec<(String, f32)> = Vec::new();
    eprintln!("clone: ranking stock voices against the reference ...");
    for (name, voice) in stock {
        if !name.starts_with('a') && !name.starts_with('b') {
            continue;
        }
        let style = style_for(voice, target_tokens.len(), &zero, env);
        let audio = rt.synth(target_tokens, style, 1.0)?;
        let trimmed = crate::trim_silence(&audio, TRIM);
        if trimmed.is_empty() {
            continue;
        }
        let wav16k = crate::resample(trimmed, NATIVE_SR, SPK_SR)?;
        let sim = dsp::cosine(&enc.embed(&wav16k)?, ref_emb);
        eprintln!("clone:   {name}: {sim:.3}");
        ranked.push((name.clone(), sim));
    }
    if ranked.is_empty() {
        bail!("no English (a*/b*) voices available to rank; pass --init explicitly");
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(AUTO_INIT_TOP);
    Ok(ranked)
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_voice(rows: usize, fill: impl Fn(usize, usize) -> f32) -> Voice {
        let mut data = Vec::with_capacity(rows * STYLE_DIM);
        for r in 0..rows {
            for c in 0..STYLE_DIM {
                data.push(fill(r, c));
            }
        }
        Voice { data, rows }
    }

    /// Regenerates the baked IPA constants. Needs espeak-ng on PATH:
    ///   cargo test regenerate_baked_ipa -- --ignored --nocapture
    #[test]
    #[ignore]
    fn regenerate_baked_ipa() {
        let target = crate::run_espeak(&crate::normalize_punctuation(REFERENCE_SCRIPT)).unwrap();
        let other = crate::run_espeak(&crate::normalize_punctuation(OTHER_TEXT)).unwrap();
        println!("const TARGET_IPA: &str = {target:?};");
        println!("const OTHER_IPA: &str = {other:?};");
    }

    #[test]
    fn voice_bin_roundtrips_through_loader() {
        let base = synthetic_voice(510, |r, c| (r as f32) * 0.001 + (c as f32) * 0.01);
        let delta: Vec<f32> = (0..STYLE_DIM).map(|c| 0.5 - (c as f32) * 0.001).collect();
        let env = Envelope {
            std: vec![1.0; STYLE_DIM],
            min: vec![f32::NEG_INFINITY; STYLE_DIM],
            max: vec![f32::INFINITY; STYLE_DIM],
        };
        let dir = std::env::temp_dir().join("storytime-clone-test");
        std::fs::create_dir_all(dir.join("voices")).unwrap();
        let path = dir.join("voices").join("testvoice.bin");
        write_voice_bin(&path, &base, &delta, &env).unwrap();
        let loaded = crate::load_voice(&dir, "testvoice").unwrap();
        assert_eq!(loaded.rows, 510);
        for r in 0..510 {
            for c in 0..STYLE_DIM {
                let want = base.data[r * STYLE_DIM + c] + delta[c];
                let got = loaded.data[r * STYLE_DIM + c];
                assert!((got - want).abs() < 1e-6, "row {r} dim {c}");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn blend_weights_and_truncates() {
        let a = synthetic_voice(511, |_, _| 1.0);
        let b = synthetic_voice(510, |_, _| 3.0);
        let blended = blend_voices(&[(a, 1.0), (b, 3.0)]);
        assert_eq!(blended.rows, 510);
        for v in &blended.data {
            assert!((v - 2.5).abs() < 1e-6); // (1*1 + 3*3) / 4
        }
    }

    #[test]
    fn envelope_and_clamping() {
        let lo = synthetic_voice(510, |_, _| -1.0);
        let hi = synthetic_voice(510, |_, _| 1.0);
        let env = envelope([&lo, &hi]);
        assert!((env.min[0] + 1.0).abs() < 1e-6);
        assert!((env.max[0] - 1.0).abs() < 1e-6);
        assert!((env.std[0] - 1.0).abs() < 1e-5);
        let base = synthetic_voice(510, |_, _| 0.5);
        let delta = vec![10.0; STYLE_DIM]; // way out of range -> clamped
        let style = style_for(&base, 42, &delta, &env);
        assert!(style.iter().all(|v| (*v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn harmonic_score_behaves() {
        // Equal components: harmonic mean == the component.
        assert!((harmonic_score(0.8, 0.8, 0.8) - 80.0).abs() < 1e-3);
        // Dominated by the weakest heavily-weighted component.
        let degenerate = harmonic_score(0.9, 0.01, 0.9);
        assert!(degenerate < 5.0, "got {degenerate}");
        // Zero-safe.
        assert!(harmonic_score(0.0, 0.0, 0.0).is_finite());
        // Feature term has little pull: big feature change, small score change.
        let a = harmonic_score(0.9, 0.9, 0.9);
        let b = harmonic_score(0.9, 0.9, 0.3);
        assert!((a - b).abs() < 5.0, "{a} vs {b}");
    }

    #[test]
    fn baked_ipa_tokenizes_under_cap() {
        // Both utterances must stay under the smallest voicepack's row cap
        // (510 rows -> 509 usable tokens) with margin. Token count is bounded
        // by the codepoint count (unknown chars like the ZWJs are dropped).
        for ipa in [TARGET_IPA, OTHER_IPA] {
            let chars = ipa.chars().filter(|c| *c != '\u{200d}').count();
            assert!((100..450).contains(&chars), "{chars} chars");
        }
    }
}

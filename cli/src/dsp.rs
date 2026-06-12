//! Audio analysis for voice cloning (`storytime clone`): WAV reading, the
//! speaker-encoder frontend (replicating Resemblyzer's preprocessing; the mel
//! filterbank is baked into spk_encoder.onnx), YIN pitch estimation, summary
//! acoustic features, and a small seedable RNG. Synthesis-side DSP
//! (resampling, trimming, fades) stays in main.rs.
//!
//! Python parity is gated by `export/verify_spk.py`: the spectrogram below is
//! fixture-tested against librosa, and the full Rust embedding must reach
//! cosine > 0.99 against the resemblyzer reference on real speech.

use std::path::Path;

use anyhow::{bail, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use realfft::RealFftPlanner;

use crate::ort_err;

/// The speaker encoder consumes 16 kHz audio (Resemblyzer convention).
pub const SPK_SR: u32 = 16_000;
/// Embedding dimensionality of the GE2E encoder.
pub const EMBED_DIM: usize = 256;

const N_FFT: usize = 400; // 25 ms @ 16 kHz
const HOP: usize = 160; // 10 ms @ 16 kHz
const N_BINS: usize = N_FFT / 2 + 1;
/// Partial-utterance windowing (resemblyzer embed_utterance, rate=1.3,
/// min_coverage=0.75): 1.6 s windows every 770 ms, averaged then re-normed.
const PARTIAL_FRAMES: usize = 160;
const FRAME_STEP: usize = 77;
const MIN_COVERAGE: f32 = 0.75;

// ---------------------------------------------------------------------------
// WAV reading
// ---------------------------------------------------------------------------

/// Read a WAV file as mono f32 in [-1, 1] plus its sample rate. Multi-channel
/// input is averaged down to mono.
pub fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("reading reference wav {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        bail!("{}: zero channels", path.display());
    }
    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            reader.samples::<f32>().collect::<Result<_, _>>()?
        }
        (hound::SampleFormat::Int, bits) if bits > 0 && bits <= 32 => {
            let peak = (1i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / peak))
                .collect::<Result<_, _>>()?
        }
        (fmt, bits) => bail!("{}: unsupported WAV format {fmt:?}/{bits}-bit", path.display()),
    };
    if channels == 1 {
        return Ok((interleaved, spec.sample_rate));
    }
    let mono = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

// ---------------------------------------------------------------------------
// Spectrogram frontend (librosa semantics; fixture-tested below)
// ---------------------------------------------------------------------------

/// Periodic Hann window, matching scipy/librosa `get_window("hann", n)`.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .map(|v| v as f32)
        .collect()
}

/// |STFT|^2 with librosa semantics: n_fft=400, hop=160, periodic hann,
/// centered (n_fft/2 zero padding on both sides, `pad_mode="constant"`).
/// Returns a flattened `[n_frames, 201]` row-major buffer and the frame count.
pub fn power_spectrogram(wav: &[f32]) -> (Vec<f32>, usize) {
    let mut padded = vec![0.0f32; wav.len() + N_FFT];
    padded[N_FFT / 2..N_FFT / 2 + wav.len()].copy_from_slice(wav);
    let n_frames = 1 + wav.len() / HOP;

    let window = hann(N_FFT);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    let mut out = Vec::with_capacity(n_frames * N_BINS);
    for f in 0..n_frames {
        let start = f * HOP;
        for (i, v) in input.iter_mut().enumerate() {
            *v = padded[start + i] * window[i];
        }
        fft.process(&mut input, &mut output)
            .expect("rfft on fixed-size buffers");
        out.extend(output.iter().map(|c| c.re * c.re + c.im * c.im));
    }
    (out, n_frames)
}

/// Resemblyzer's `normalize_volume(wav, -30, increase_only=True)`: bring the
/// signal up to -30 dBFS RMS, never attenuating.
pub fn normalize_volume(wav: &mut [f32]) {
    let mean_sq = wav.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>() / wav.len().max(1) as f64;
    if mean_sq <= 0.0 {
        return;
    }
    let change = -30.0 - 10.0 * mean_sq.log10();
    if change < 0.0 {
        return;
    }
    let gain = 10f64.powf(change / 20.0) as f32;
    for s in wav {
        *s *= gain;
    }
}

/// Resemblyzer's `compute_partial_slices`: start frames of the 160-frame
/// partial windows, plus the wav length (in samples) the signal must be
/// zero-padded to before computing the spectrogram.
fn partial_slices(n_samples: usize) -> (Vec<usize>, usize) {
    let n_frames = (n_samples + 1).div_ceil(HOP);
    let steps = (n_frames as i64 - PARTIAL_FRAMES as i64 + FRAME_STEP as i64 + 1).max(1) as usize;
    let mut starts: Vec<usize> = (0..steps).step_by(FRAME_STEP).collect();
    let last = *starts.last().unwrap();
    let coverage =
        (n_samples as f32 - (last * HOP) as f32) / (PARTIAL_FRAMES * HOP) as f32;
    if coverage < MIN_COVERAGE && starts.len() > 1 {
        starts.pop();
    }
    let padded_len = (starts.last().unwrap() + PARTIAL_FRAMES) * HOP;
    (starts, padded_len)
}

// ---------------------------------------------------------------------------
// Speaker encoder (GE2E via ONNX Runtime, CPU)
// ---------------------------------------------------------------------------

pub struct SpeakerEncoder {
    session: Session,
}

impl SpeakerEncoder {
    pub fn load(assets: &Path) -> Result<Self> {
        let path = assets.join("spk_encoder.onnx");
        if !path.exists() {
            bail!(
                "{} not found — re-run ./setup.sh (or: python export/export.py \
                 --skip-model --skip-voices) to export the speaker encoder",
                path.display()
            );
        }
        // Safe to call alongside Runtime::init: committing the ort environment
        // is idempotent for our purposes. The encoder is a 1.4 M-param LSTM;
        // plain CPU execution is plenty.
        ort::init().with_name("storytime").commit().map_err(ort_err)?;
        let session = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .commit_from_file(&path)
            .map_err(ort_err)?;
        Ok(SpeakerEncoder { session })
    }

    /// Embed a 16 kHz mono utterance: L2-normalized 256-d speaker embedding
    /// (mean of partial-window embeddings, re-normalized — resemblyzer's
    /// `embed_utterance`, minus its webrtcvad trim).
    pub fn embed(&mut self, wav16k: &[f32]) -> Result<[f32; EMBED_DIM]> {
        if wav16k.is_empty() {
            bail!("cannot embed empty audio");
        }
        let mut wav = wav16k.to_vec();
        normalize_volume(&mut wav);
        let (starts, padded_len) = partial_slices(wav.len());
        if padded_len > wav.len() {
            wav.resize(padded_len, 0.0);
        }
        let (spec, n_frames) = power_spectrogram(&wav);
        debug_assert!(starts.iter().all(|s| s + PARTIAL_FRAMES <= n_frames));

        let mut batch = Vec::with_capacity(starts.len() * PARTIAL_FRAMES * N_BINS);
        for &s in &starts {
            batch.extend_from_slice(&spec[s * N_BINS..(s + PARTIAL_FRAMES) * N_BINS]);
        }
        let tensor = Tensor::from_array((
            vec![starts.len() as i64, PARTIAL_FRAMES as i64, N_BINS as i64],
            batch,
        ))
        .map_err(ort_err)?;
        let outputs = self
            .session
            .run(ort::inputs!["power_spec" => tensor])
            .map_err(ort_err)?;
        let (_shape, partials) = outputs["embedding"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;

        // L2-normalizing the mean of partials == L2-normalizing their sum.
        let mut mean = [0f32; EMBED_DIM];
        for p in 0..starts.len() {
            for (d, m) in mean.iter_mut().enumerate() {
                *m += partials[p * EMBED_DIM + d];
            }
        }
        let norm = mean.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for m in &mut mean {
            *m /= norm;
        }
        Ok(mean)
    }
}

/// Cosine similarity. GE2E embeddings are non-negative and L2-normalized, so
/// for them this lands in [0, 1].
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

// ---------------------------------------------------------------------------
// YIN pitch estimation
// ---------------------------------------------------------------------------

const YIN_WINDOW: usize = 1024;
const YIN_HOP: usize = 1024;
const YIN_THRESHOLD: f32 = 0.1;
const F0_MIN: f32 = 50.0;
const F0_MAX: f32 = 500.0;

/// Per-frame fundamental frequency via YIN (difference function + cumulative
/// mean normalization + parabolic interpolation). `None` = unvoiced frame.
pub fn yin_f0(wav: &[f32], sr: u32) -> Vec<Option<f32>> {
    let tau_min = (sr as f32 / F0_MAX).floor() as usize;
    let tau_max = (sr as f32 / F0_MIN).ceil() as usize;
    let frame_len = YIN_WINDOW + tau_max;
    let mut out = Vec::new();
    if wav.len() < frame_len {
        return out;
    }

    let mut d = vec![0f32; tau_max + 1];
    let mut cmndf = vec![0f32; tau_max + 1];
    let mut start = 0;
    while start + frame_len <= wav.len() {
        let x = &wav[start..start + frame_len];
        for tau in 1..=tau_max {
            let mut sum = 0f32;
            for j in 0..YIN_WINDOW {
                let diff = x[j] - x[j + tau];
                sum += diff * diff;
            }
            d[tau] = sum;
        }
        let mut running = 0f32;
        cmndf[0] = 1.0;
        for tau in 1..=tau_max {
            running += d[tau];
            cmndf[tau] = if running > 0.0 {
                d[tau] * tau as f32 / running
            } else {
                1.0
            };
        }
        let mut f0 = None;
        let mut tau = tau_min.max(2);
        while tau < tau_max {
            if cmndf[tau] < YIN_THRESHOLD {
                // Descend to the local minimum, then refine parabolically.
                while tau + 1 < tau_max && cmndf[tau + 1] < cmndf[tau] {
                    tau += 1;
                }
                let (a, b, c) = (cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]);
                let denom = a + c - 2.0 * b;
                let offset = if denom.abs() > 1e-12 { 0.5 * (a - c) / denom } else { 0.0 };
                f0 = Some(sr as f32 / (tau as f32 + offset.clamp(-1.0, 1.0)));
                break;
            }
            tau += 1;
        }
        out.push(f0);
        start += YIN_HOP;
    }
    out
}

// ---------------------------------------------------------------------------
// Summary acoustic features (the low-weight guard-rail score term)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct AcousticStats {
    pub f0_mean: f32,
    pub f0_std: f32,
    pub voiced_ratio: f32,
    pub rms_mean: f32,
    pub rms_std: f32,
    pub centroid_mean: f32,
    pub centroid_std: f32,
    pub rolloff_mean: f32,
    pub zcr_mean: f32,
}

fn mean_std(values: impl Iterator<Item = f32> + Clone) -> (f32, f32) {
    let (mut n, mut sum) = (0usize, 0f64);
    for v in values.clone() {
        n += 1;
        sum += v as f64;
    }
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = sum / n as f64;
    let var = values.map(|v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
    (mean as f32, var.sqrt() as f32)
}

/// Summarize a 16 kHz utterance: pitch statistics, frame energy, spectral
/// centroid / rolloff, zero-crossing rate.
pub fn acoustic_stats(wav: &[f32], sr: u32) -> AcousticStats {
    let mut stats = AcousticStats::default();
    if wav.is_empty() {
        return stats;
    }

    let f0s = yin_f0(wav, sr);
    let voiced: Vec<f32> = f0s.iter().flatten().copied().collect();
    if !f0s.is_empty() {
        stats.voiced_ratio = voiced.len() as f32 / f0s.len() as f32;
    }
    if !voiced.is_empty() {
        let (m, s) = mean_std(voiced.iter().copied());
        stats.f0_mean = m;
        stats.f0_std = s;
    }

    let (spec, n_frames) = power_spectrogram(wav);
    let hz_per_bin = sr as f32 / N_FFT as f32;
    let mut centroids = Vec::with_capacity(n_frames);
    let mut rolloffs = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        let frame = &spec[f * N_BINS..(f + 1) * N_BINS];
        let total: f32 = frame.iter().sum();
        if total <= 1e-10 {
            continue;
        }
        let weighted: f32 = frame
            .iter()
            .enumerate()
            .map(|(k, p)| k as f32 * hz_per_bin * p)
            .sum();
        centroids.push(weighted / total);
        let mut acc = 0f32;
        let mut roll = (N_BINS - 1) as f32 * hz_per_bin;
        for (k, p) in frame.iter().enumerate() {
            acc += p;
            if acc >= 0.85 * total {
                roll = k as f32 * hz_per_bin;
                break;
            }
        }
        rolloffs.push(roll);
    }
    (stats.centroid_mean, stats.centroid_std) = mean_std(centroids.iter().copied());
    (stats.rolloff_mean, _) = mean_std(rolloffs.iter().copied());

    let rms: Vec<f32> = wav
        .chunks(YIN_HOP)
        .map(|c| (c.iter().map(|s| s * s).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    (stats.rms_mean, stats.rms_std) = mean_std(rms.iter().copied());

    let crossings = wav.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count();
    stats.zcr_mean = crossings as f32 / wav.len() as f32;

    stats
}

/// Similarity in (0, 1]: mean per-feature relative difference against the
/// reference, each capped at 2.0 (NaN-safe: division floors at an epsilon, so
/// features that are legitimately ~0 in the reference don't blow up — the
/// upstream KVoiceWalk formula divides by raw target values and can go inf).
pub fn feature_similarity(candidate: &AcousticStats, reference: &AcousticStats) -> f32 {
    let pairs = [
        (candidate.f0_mean, reference.f0_mean),
        (candidate.f0_std, reference.f0_std),
        (candidate.voiced_ratio, reference.voiced_ratio),
        (candidate.rms_mean, reference.rms_mean),
        (candidate.rms_std, reference.rms_std),
        (candidate.centroid_mean, reference.centroid_mean),
        (candidate.centroid_std, reference.centroid_std),
        (candidate.rolloff_mean, reference.rolloff_mean),
        (candidate.zcr_mean, reference.zcr_mean),
    ];
    let mean_diff = pairs
        .iter()
        .map(|(c, r)| ((c - r).abs() / r.abs().max(1e-6)).min(2.0))
        .sum::<f32>()
        / pairs.len() as f32;
    (1.0 - mean_diff / 2.0).max(0.01)
}

// ---------------------------------------------------------------------------
// Seedable RNG (SplitMix64 + Box-Muller; avoids a `rand` dependency)
// ---------------------------------------------------------------------------

pub struct Rng {
    state: u64,
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed, spare: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [lo, hi).
    pub fn uniform_in(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal via Box-Muller.
    pub fn gauss(&mut self) -> f32 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = self.uniform().max(f32::MIN_POSITIVE);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture signal: two exact-bin tones, computed in f64 then cast,
    /// matching export/verify_spk.py --dump-fixture.
    fn fixture_signal() -> Vec<f32> {
        (0..800)
            .map(|n| {
                let t = n as f64 / SPK_SR as f64;
                (0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 1000.0 * t + 0.5).sin())
                    as f32
            })
            .collect()
    }

    // Generated by export/verify_spk.py --dump-fixture (6 frames).
    const FIXTURE_FRAME_0: [f32; 201] = [1.223558e+01, 1.259700e+01, 1.296399e+01, 1.426728e+01, 1.563324e+01, 1.897553e+01, 2.264150e+01, 3.346598e+01, 4.639856e+01, 1.463630e+02, 4.548015e+02, 6.186984e+02, 3.603428e+02, 8.341689e+01, 1.457248e+01, 7.405134e+00, 2.677135e+00, 1.281868e+00, 4.804881e-01, 3.330463e-01, 5.815496e-01, 1.781526e+00, 3.850921e+00, 2.192645e+01, 9.137065e+01, 1.519349e+02, 1.074860e+02, 3.238322e+01, 8.969713e+00, 5.910304e+00, 3.486759e+00, 2.604498e+00, 1.850712e+00, 1.472228e+00, 1.137033e+00, 9.393589e-01, 7.605966e-01, 6.445061e-01, 5.380737e-01, 4.643752e-01, 3.961480e-01, 3.466892e-01, 3.005674e-01, 2.659612e-01, 2.335057e-01, 2.084890e-01, 1.849193e-01, 1.663530e-01, 1.487945e-01, 1.347132e-01, 1.213539e-01, 1.104772e-01, 1.001301e-01, 9.159625e-02, 8.345897e-02, 7.667177e-02, 7.018673e-02, 6.472404e-02, 5.949511e-02, 5.505179e-02, 5.079189e-02, 4.714354e-02, 4.364078e-02, 4.061959e-02, 3.771526e-02, 3.519402e-02, 3.276756e-02, 3.064889e-02, 2.860775e-02, 2.681584e-02, 2.508787e-02, 2.356334e-02, 2.209196e-02, 2.078786e-02, 1.952830e-02, 1.840713e-02, 1.732350e-02, 1.635505e-02, 1.541841e-02, 1.457819e-02, 1.376512e-02, 1.303321e-02, 1.232457e-02, 1.168455e-02, 1.106460e-02, 1.050291e-02, 9.958591e-03, 9.463998e-03, 8.984525e-03, 8.547594e-03, 8.123868e-03, 7.736736e-03, 7.361207e-03, 7.017202e-03, 6.683423e-03, 6.376948e-03, 6.079486e-03, 5.805707e-03, 5.539949e-03, 5.294807e-03, 5.056824e-03, 4.836843e-03, 4.623238e-03, 4.425369e-03, 4.233212e-03, 4.054868e-03, 3.881670e-03, 3.720613e-03, 3.564195e-03, 3.418474e-03, 3.276942e-03, 3.144851e-03, 3.016572e-03, 2.896649e-03, 2.780194e-03, 2.671138e-03, 2.565234e-03, 2.465898e-03, 2.369455e-03, 2.278850e-03, 2.190887e-03, 2.108123e-03, 2.027784e-03, 1.952083e-03, 1.878625e-03, 1.809299e-03, 1.742041e-03, 1.678485e-03, 1.616844e-03, 1.558509e-03, 1.501950e-03, 1.448351e-03, 1.396402e-03, 1.347110e-03, 1.299357e-03, 1.253986e-03, 1.210048e-03, 1.168251e-03, 1.127796e-03, 1.089264e-03, 1.051988e-03, 1.016439e-03, 9.820788e-04, 9.492697e-04, 9.175770e-04, 8.872836e-04, 8.580422e-04, 8.300614e-04, 8.030719e-04, 7.772102e-04, 7.522961e-04, 7.284080e-04, 7.054089e-04, 6.833268e-04, 6.621006e-04, 6.417022e-04, 6.221153e-04, 6.032718e-04, 5.852056e-04, 5.678115e-04, 5.511583e-04, 5.351098e-04, 5.197792e-04, 5.049880e-04, 4.908795e-04, 4.772684e-04, 4.643154e-04, 4.518007e-04, 4.399246e-04, 4.284459e-04, 4.175843e-04, 4.070818e-04, 3.971733e-04, 3.875864e-04, 3.785809e-04, 3.698655e-04, 3.617164e-04, 3.538287e-04, 3.464883e-04, 3.393896e-04, 3.328311e-04, 3.264841e-04, 3.206624e-04, 3.150355e-04, 3.099264e-04, 3.050010e-04, 3.005809e-04, 2.963195e-04, 2.925628e-04, 2.889577e-04, 2.858437e-04, 2.828687e-04, 2.803842e-04, 2.780356e-04, 2.761733e-04, 2.744352e-04, 2.731778e-04, 2.720421e-04, 2.713841e-04, 2.708469e-04, 2.707904e-04];
    const FIXTURE_FRAME_3: [f32; 201] = [1.504585e-14, 1.735618e-14, 1.828171e-14, 2.022731e-14, 7.954110e-15, 8.107503e-15, 6.160503e-15, 2.537414e-14, 1.543359e-14, 9.060450e-15, 6.249999e+02, 2.500000e+03, 6.249999e+02, 3.488243e-14, 1.860011e-14, 8.574804e-15, 7.564620e-16, 6.754115e-15, 2.923802e-15, 3.560003e-14, 3.490529e-14, 3.808320e-14, 1.805045e-14, 8.713734e-15, 1.562500e+02, 6.250000e+02, 1.562500e+02, 3.910468e-14, 1.740978e-14, 1.301815e-14, 7.521999e-15, 4.736412e-15, 7.256520e-15, 3.264509e-14, 3.618479e-14, 4.866263e-14, 9.694468e-15, 6.435758e-16, 6.024404e-15, 2.083461e-14, 2.142082e-15, 2.951472e-15, 1.872988e-15, 4.309962e-15, 1.726454e-15, 1.898772e-14, 1.233514e-14, 7.335118e-15, 1.266861e-14, 8.469610e-14, 2.503231e-14, 4.736101e-15, 1.050088e-14, 1.902041e-14, 1.976133e-14, 2.729052e-14, 3.668685e-15, 1.128011e-14, 1.591932e-14, 2.157091e-14, 1.333736e-14, 1.475929e-14, 7.877949e-15, 4.064674e-14, 3.274231e-14, 5.047581e-14, 4.619892e-14, 5.843416e-14, 2.079303e-14, 4.081003e-15, 5.288247e-15, 7.302860e-15, 1.827006e-14, 3.759060e-14, 2.007024e-14, 2.645485e-14, 1.026023e-14, 2.880670e-14, 2.947135e-14, 3.399927e-14, 3.119897e-15, 5.306473e-15, 1.033123e-14, 1.752381e-14, 5.884099e-15, 9.528717e-15, 1.217528e-14, 1.573580e-14, 9.931900e-15, 5.912046e-15, 9.326887e-15, 1.439234e-14, 9.981529e-15, 2.043122e-14, 1.331809e-14, 5.872467e-14, 2.382118e-14, 4.577055e-15, 1.161287e-14, 6.606924e-14, 2.906977e-14, 7.273128e-15, 1.458732e-14, 4.197855e-14, 5.285308e-15, 1.059378e-14, 7.210191e-16, 7.619136e-15, 1.695757e-15, 2.067738e-15, 1.817176e-14, 5.493147e-14, 1.319073e-14, 6.144852e-14, 8.583676e-15, 4.548736e-15, 1.055382e-15, 9.400085e-15, 1.054917e-14, 1.874062e-14, 5.371925e-16, 2.863131e-14, 1.160085e-14, 1.391337e-14, 2.359065e-15, 2.406539e-14, 6.266564e-15, 1.000356e-14, 1.120294e-15, 1.707642e-15, 4.644776e-16, 1.984148e-15, 5.889470e-17, 2.433153e-15, 5.579329e-15, 3.923136e-14, 1.946725e-16, 3.799527e-14, 1.245404e-15, 3.554141e-14, 1.605988e-14, 2.439169e-14, 1.353350e-15, 2.216954e-14, 2.322936e-15, 5.689486e-15, 2.862482e-15, 2.403676e-14, 1.986900e-14, 2.936417e-14, 1.534524e-14, 5.594856e-14, 1.310233e-14, 6.546751e-15, 1.068974e-15, 4.052607e-15, 1.883286e-15, 9.550479e-15, 1.177444e-15, 3.084380e-15, 8.978311e-15, 5.003430e-14, 1.142879e-14, 6.221372e-14, 3.072473e-14, 2.528822e-14, 3.591308e-15, 5.664084e-15, 4.586336e-16, 3.178726e-15, 5.211594e-15, 1.061351e-14, 6.821477e-15, 1.043590e-14, 1.034282e-15, 2.671380e-14, 2.592452e-15, 2.274333e-14, 2.454364e-14, 2.725420e-14, 1.182453e-14, 3.612557e-15, 9.320379e-16, 3.615649e-15, 2.849640e-14, 8.996152e-14, 2.941311e-14, 1.411959e-14, 9.934747e-15, 2.763478e-14, 1.576558e-14, 3.442705e-14, 1.790892e-14, 7.737554e-15, 1.233815e-14, 2.692222e-14, 2.371959e-15, 4.706973e-14, 5.192242e-15, 2.806173e-14, 2.639227e-14];

    #[test]
    fn hann_is_periodic() {
        let w = hann(400);
        assert!((w[0]).abs() < 1e-7);
        assert!((w[100] - 0.5).abs() < 1e-6);
        assert!((w[200] - 1.0).abs() < 1e-6);
        // Periodic (fftbins=True): w[n] == w[N-n], and w[N-1] != 0.
        assert!((w[150] - w[250]).abs() < 1e-6);
        assert!(w[399] > 0.0);
    }

    #[test]
    fn spectrogram_matches_librosa_fixture() {
        let (spec, n_frames) = power_spectrogram(&fixture_signal());
        assert_eq!(n_frames, 6);
        for (frame_idx, expected) in [(0usize, &FIXTURE_FRAME_0), (3, &FIXTURE_FRAME_3)] {
            let frame = &spec[frame_idx * N_BINS..(frame_idx + 1) * N_BINS];
            let peak = expected.iter().cloned().fold(0f32, f32::max);
            for (k, (&got, &want)) in frame.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-4 * peak,
                    "frame {frame_idx} bin {k}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn spectrogram_peaks_at_tone_bin() {
        // 440 Hz at 16 kHz with n_fft=400 -> bin 11 exactly.
        let wav: Vec<f32> = (0..3200)
            .map(|n| (2.0 * std::f32::consts::PI * 440.0 * n as f32 / 16_000.0).sin())
            .collect();
        let (spec, n_frames) = power_spectrogram(&wav);
        let mid = n_frames / 2;
        let frame = &spec[mid * N_BINS..(mid + 1) * N_BINS];
        let argmax = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(argmax, 11);
    }

    #[test]
    fn partial_slices_match_resemblyzer() {
        // 160 frames exactly: the second window covers only ~52% -> dropped.
        let (starts, padded) = partial_slices(160 * HOP);
        assert_eq!(starts, vec![0]);
        assert_eq!(padded, 160 * HOP);
        // 10 s @ 16 kHz: ceil(160001/160)=1001 frames, steps=919 -> 12 windows;
        // last start 847, coverage (160000-135520)/25600 ~ 0.96 -> kept.
        let (starts, padded) = partial_slices(160_000);
        assert_eq!(starts.len(), 12);
        assert_eq!(*starts.last().unwrap(), 847);
        assert_eq!(padded, (847 + 160) * HOP);
        // Tiny input: a single, fully padded window.
        let (starts, padded) = partial_slices(100);
        assert_eq!(starts, vec![0]);
        assert_eq!(padded, PARTIAL_FRAMES * HOP);
    }

    #[test]
    fn yin_finds_sine_frequencies() {
        for hz in [220.0f32, 440.0] {
            let wav: Vec<f32> = (0..16_000)
                .map(|n| (2.0 * std::f32::consts::PI * hz * n as f32 / 16_000.0).sin())
                .collect();
            let f0s = yin_f0(&wav, 16_000);
            assert!(!f0s.is_empty());
            for f0 in f0s.iter().flatten() {
                assert!((f0 - hz).abs() < 2.0, "expected ~{hz} Hz, got {f0}");
            }
            assert!(f0s.iter().all(|f| f.is_some()), "{hz} Hz sine should be voiced");
        }
        // Silence is unvoiced.
        let silent = vec![0f32; 16_000];
        assert!(yin_f0(&silent, 16_000).iter().all(|f| f.is_none()));
    }

    #[test]
    fn normalize_volume_is_increase_only() {
        // A quiet signal is brought up to -30 dBFS.
        let mut quiet: Vec<f32> = (0..16_000)
            .map(|n| 0.001 * (2.0 * std::f32::consts::PI * 220.0 * n as f32 / 16_000.0).sin())
            .collect();
        normalize_volume(&mut quiet);
        let mean_sq: f64 =
            quiet.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>() / quiet.len() as f64;
        let dbfs = 10.0 * mean_sq.log10();
        assert!((dbfs + 30.0).abs() < 0.1, "got {dbfs} dBFS");
        // A loud signal is left alone.
        let loud: Vec<f32> = (0..16_000)
            .map(|n| 0.9 * (2.0 * std::f32::consts::PI * 220.0 * n as f32 / 16_000.0).sin())
            .collect();
        let mut loud2 = loud.clone();
        normalize_volume(&mut loud2);
        assert_eq!(loud, loud2);
        // Silence is a no-op, not a NaN.
        let mut silence = vec![0f32; 100];
        normalize_volume(&mut silence);
        assert!(silence.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn feature_similarity_basics() {
        let a = acoustic_stats(
            &(0..32_000)
                .map(|n| (2.0 * std::f32::consts::PI * 220.0 * n as f32 / 16_000.0).sin())
                .collect::<Vec<_>>(),
            16_000,
        );
        assert!((feature_similarity(&a, &a) - 1.0).abs() < 1e-6);
        let zero = AcousticStats::default();
        let sim = feature_similarity(&a, &zero);
        assert!(sim.is_finite() && sim >= 0.01, "NaN-safety: got {sim}");
    }

    /// Full-pipeline parity against the Python frontend. Run manually:
    ///   storytime ... -o /tmp/spk_parity.wav     # any clean speech, any rate
    ///   python export/verify_spk.py /tmp/spk_parity.wav \
    ///       --dump-embedding /tmp/spk_parity_emb.json
    ///   cargo test spk_embedding -- --ignored
    /// Ignored by default: needs assets/spk_encoder.onnx and the files above.
    #[test]
    #[ignore]
    fn spk_embedding_matches_python_frontend() {
        let assets = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("assets");
        let (wav, sr) = read_wav_mono(Path::new("/tmp/spk_parity.wav")).unwrap();
        let wav16k = crate::resample(&wav, sr, SPK_SR).unwrap();
        let ours = SpeakerEncoder::load(&assets).unwrap().embed(&wav16k).unwrap();
        let json = std::fs::read_to_string("/tmp/spk_parity_emb.json").unwrap();
        let python: Vec<f32> = serde_json::from_str(&json).unwrap();
        let cos = cosine(&ours, &python);
        assert!(cos > 0.999, "Rust-vs-Python embedding cosine too low: {cos}");
    }

    #[test]
    fn rng_is_deterministic_and_roughly_normal() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..5 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut r = Rng::new(7);
        let n = 20_000;
        let draws: Vec<f32> = (0..n).map(|_| r.gauss()).collect();
        let mean = draws.iter().sum::<f32>() / n as f32;
        let var = draws.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
        for _ in 0..100 {
            let u = r.uniform_in(0.01, 0.15);
            assert!((0.01..0.15).contains(&u));
        }
    }
}

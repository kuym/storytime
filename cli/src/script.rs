//! Screenplay / multi-voice script parsing and voice resolution.
//!
//! Script mode (`--script`) reads a screenplay in the universal `NAME: dialogue`
//! convention that every LLM already produces reliably, with an optional cast
//! header that assigns each character a voice — either an explicit Kokoro voice
//! id (`af_bella`) or a trait list (`female, american, young`) that is resolved
//! to a concrete voice here.
//!
//! Format (all matching is case-insensitive and whitespace-tolerant):
//!
//! ```text
//! # Cast
//! ALICE: female, american, young
//! BOB: male, british, gruff
//! NARRATOR: af_heart            # explicit voice id also allowed
//!
//! ---
//!
//! NARRATOR: They stood at the door.
//! ALICE: Are you sure about this? I really think we should--
//! BOB: Stop. We're going in.
//! ```
//!
//! - The **cast block** is the lines under a `Cast` / `Dramatis Personae`
//!   heading, ending at the next heading or `---` rule. It is never spoken.
//! - A **speech** starts at a `NAME:` line whose `NAME` is a declared character
//!   or is written in screenplay all-caps; the remainder plus any following
//!   non-speaker lines (until a blank line or the next speaker) is that
//!   character's text.
//! - Lines with no speaker are **narration**, spoken by `NARRATOR` (or the
//!   `--narrator` / `--voice` default).
//! - A speech whose text ends in `--` / `—` is **interrupted**: the next speech
//!   overlaps it.
//! - `(parentheticals)` are stage directions and are stripped, not spoken.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};

/// How a cast entry names its voice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceSpec {
    /// An explicit Kokoro voice id, e.g. `af_bella`.
    Explicit(String),
    /// A free-form trait list, e.g. `["female", "american", "young"]`.
    Traits(Vec<String>),
}

/// One declared character and the voice it requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CastEntry {
    /// Display name as written (e.g. `ALICE`).
    pub name: String,
    pub spec: VoiceSpec,
}

/// One continuous turn of speech by a single character.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Speech {
    /// Lowercased character key used for voice lookup.
    pub character: String,
    /// Spoken text (markdown preserved; parentheticals stripped).
    pub text: String,
    /// True when this speech interrupts (overlaps) the previous one — set when
    /// the previous speech ended on a `--` / `—` cue.
    pub interrupts_prev: bool,
}

/// A parsed script: its cast declarations and ordered speeches.
#[derive(Debug)]
pub struct Script {
    pub cast: Vec<CastEntry>,
    pub speeches: Vec<Speech>,
}

/// Lowercased lookup key for a character name.
fn key(name: &str) -> String {
    name.trim().to_lowercase()
}

/// True if `s` looks like a Kokoro voice id: two lowercase letters, `_`, then
/// lowercase alphanumerics/underscores (e.g. `af_heart`, `bm_george`).
fn is_voice_id(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 4 || b[2] != b'_' {
        return false;
    }
    b[0].is_ascii_lowercase()
        && b[1].is_ascii_lowercase()
        && b[3..]
            .iter()
            .all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
}

/// True if a line's pre-colon label should be read as a speaker name: either it
/// matches a declared character, or it is written in screenplay all-caps (no
/// lowercase letters, has a letter, reasonably short). The all-caps rule keeps
/// ordinary prose like `He said: hello` from being misread as a speaker.
fn is_speaker_label(name: &str, cast_keys: &HashSet<String>) -> bool {
    let n = name.trim();
    if n.is_empty() || n.len() > 32 {
        return false;
    }
    if cast_keys.contains(&key(n)) {
        return true;
    }
    let has_alpha = n.chars().any(|c| c.is_alphabetic());
    let no_lower = !n.chars().any(|c| c.is_lowercase());
    has_alpha && no_lower
}

/// Heading title (text after the leading `#`s), if `line` is an ATX heading.
fn heading_title(line: &str) -> Option<String> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let title = t.trim_start_matches('#').trim();
    Some(title.to_string())
}

/// True for a thematic-break / horizontal rule line (`---`, `***`, `___`).
fn is_rule(line: &str) -> bool {
    let t = line.trim();
    t.len() >= 3
        && (t.chars().all(|c| c == '-')
            || t.chars().all(|c| c == '*')
            || t.chars().all(|c| c == '_'))
}

/// Strip a leading markdown list bullet (`- `, `* `, `+ `, `1. `) if present.
fn strip_bullet(line: &str) -> &str {
    let t = line.trim_start();
    for p in ["- ", "* ", "+ "] {
        if let Some(r) = t.strip_prefix(p) {
            return r;
        }
    }
    // numbered: digits then `. `
    let bytes = t.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i > 0 && t[i..].starts_with(". ") {
        return &t[i + 2..];
    }
    t
}

/// Parse one `NAME: spec` (or bare `NAME`) cast line into an entry.
fn parse_cast_line(line: &str) -> Option<CastEntry> {
    let line = strip_bullet(line).trim();
    if line.is_empty() {
        return None;
    }
    // Drop trailing `# comment` so `NAME: af_heart   # explicit` works.
    let line = match line.find('#') {
        Some(i) => line[..i].trim(),
        None => line,
    };
    if line.is_empty() {
        return None;
    }
    match line.split_once(':') {
        Some((name, spec)) => {
            let name = name.trim();
            if name.is_empty() {
                return None;
            }
            Some(CastEntry {
                name: name.to_string(),
                spec: parse_spec(spec.trim()),
            })
        }
        // Bare name with no spec → fully auto-assigned.
        None => Some(CastEntry {
            name: line.to_string(),
            spec: VoiceSpec::Traits(Vec::new()),
        }),
    }
}

/// Parse the voice spec after a cast entry's colon.
fn parse_spec(spec: &str) -> VoiceSpec {
    if spec.is_empty() {
        return VoiceSpec::Traits(Vec::new());
    }
    if !spec.contains(',') && !spec.contains(char::is_whitespace) && is_voice_id(spec) {
        return VoiceSpec::Explicit(spec.to_string());
    }
    let traits: Vec<String> = spec
        .split([',', ' ', '\t'])
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    VoiceSpec::Traits(traits)
}

/// Parse a standalone cast section (an optional `--cast` file): every non-blank,
/// non-heading, non-rule line is treated as a cast entry.
fn parse_cast_section(text: &str) -> Vec<CastEntry> {
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() || heading_title(line).is_some() || is_rule(line) {
            continue;
        }
        if let Some(e) = parse_cast_line(line) {
            out.push(e);
        }
    }
    out
}

/// Locate an inline cast block in `lines`. Returns its entries and the
/// half-open line range `[start, end)` to exclude from speech parsing.
fn find_inline_cast(lines: &[&str]) -> (Vec<CastEntry>, Option<(usize, usize)>) {
    let mut head: Option<usize> = None;
    for (i, line) in lines.iter().enumerate() {
        if let Some(title) = heading_title(line) {
            let t = title.to_lowercase();
            if t == "cast" || t == "dramatis personae" {
                head = Some(i);
                break;
            }
        }
    }
    let Some(start) = head else {
        return (Vec::new(), None);
    };
    // Block runs from the line after the heading to the next heading or rule.
    let mut end = lines.len();
    for (j, line) in lines.iter().enumerate().skip(start + 1) {
        if heading_title(line).is_some() || is_rule(line) {
            end = j;
            break;
        }
    }
    let mut entries = Vec::new();
    for line in &lines[start + 1..end] {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(e) = parse_cast_line(line) {
            entries.push(e);
        }
    }
    (entries, Some((start, end)))
}

/// Remove `(parenthetical)` stage directions from a speech line.
fn strip_parentheticals(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0u32;
    for c in s.chars() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
            }
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    // Collapse any double spaces left behind by an inline removal.
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// True if a finished speech ends on an interruption cue (`--`, `—`, or `–`).
fn ends_with_interruption(text: &str) -> bool {
    let t = text.trim_end();
    t.ends_with("--") || t.ends_with('\u{2014}') || t.ends_with('\u{2013}')
}

/// Normalize a trailing `--` interruption cue to an em-dash so Kokoro renders
/// the trained cut-off prosody; leaves the rest of the text untouched.
fn normalize_interruption_cue(text: &str) -> String {
    let t = text.trim_end();
    if t.ends_with("--") {
        let kept = t.trim_end_matches('-');
        format!("{kept}\u{2014}")
    } else {
        t.to_string()
    }
}

/// Parse a screenplay. `body` is the main input; `extra_cast` is the optional
/// contents of a separate `--cast` file (its entries take precedence on name
/// conflicts). Returns the cast and the ordered list of speeches.
pub fn parse(body: &str, extra_cast: Option<&str>) -> Result<Script> {
    let lines: Vec<&str> = body.lines().collect();
    let (inline_cast, skip) = find_inline_cast(&lines);

    // Combine cast sources: the explicit --cast file wins on name conflicts.
    let mut cast: Vec<CastEntry> = Vec::new();
    let mut have: HashSet<String> = HashSet::new();
    if let Some(extra) = extra_cast {
        for e in parse_cast_section(extra) {
            if have.insert(key(&e.name)) {
                cast.push(e);
            }
        }
    }
    for e in inline_cast {
        if have.insert(key(&e.name)) {
            cast.push(e);
        }
    }

    let cast_keys: HashSet<String> = cast.iter().map(|e| key(&e.name)).collect();

    // Parse speeches from the body, skipping the inline cast block.
    let mut speeches: Vec<Speech> = Vec::new();
    let mut cur: Option<Speech> = None;
    let mut prev_interrupt = false;

    let flush = |cur: &mut Option<Speech>,
                 speeches: &mut Vec<Speech>,
                 prev_interrupt: &mut bool| {
        if let Some(mut sp) = cur.take() {
            sp.text = sp.text.trim().to_string();
            if sp.text.is_empty() {
                return;
            }
            *prev_interrupt = ends_with_interruption(&sp.text);
            sp.text = normalize_interruption_cue(&sp.text);
            speeches.push(sp);
        }
    };

    for (i, raw) in lines.iter().enumerate() {
        if let Some((s, e)) = skip {
            if i >= s && i < e {
                continue;
            }
        }
        let line = *raw;
        // Blank lines, headings, and horizontal rules are structural breaks:
        // they end the current speech and are never spoken.
        if line.trim().is_empty() || is_rule(line) || heading_title(line).is_some() {
            flush(&mut cur, &mut speeches, &mut prev_interrupt);
            continue;
        }

        // Is this a speaker line? Only treat the part before the FIRST colon as
        // a label, and only if it qualifies (declared or all-caps).
        let speaker = line.split_once(':').and_then(|(name, rest)| {
            if is_speaker_label(name, &cast_keys) {
                Some((key(name), rest))
            } else {
                None
            }
        });

        if let Some((character, rest)) = speaker {
            flush(&mut cur, &mut speeches, &mut prev_interrupt);
            let text = strip_parentheticals(rest);
            cur = Some(Speech {
                character,
                text,
                interrupts_prev: prev_interrupt,
            });
            prev_interrupt = false;
        } else {
            let text = strip_parentheticals(line);
            if text.is_empty() {
                continue;
            }
            match &mut cur {
                Some(sp) => {
                    sp.text.push(' ');
                    sp.text.push_str(&text);
                }
                None => {
                    cur = Some(Speech {
                        character: "narrator".to_string(),
                        text,
                        interrupts_prev: prev_interrupt,
                    });
                    prev_interrupt = false;
                }
            }
        }
    }
    flush(&mut cur, &mut speeches, &mut prev_interrupt);

    if speeches.is_empty() {
        bail!("no speeches found — expected `NAME: dialogue` lines");
    }
    Ok(Script { cast, speeches })
}

// ----------------------------------------------------------------------------
// Voice resolution
// ----------------------------------------------------------------------------

/// (language, gender) decoded from a voice id's two-letter prefix, e.g.
/// `af_heart` → (`'a'`, `'f'`).
fn lang_gender(voice_id: &str) -> Option<(char, char)> {
    let b = voice_id.as_bytes();
    if b.len() >= 3 && b[2] == b'_' {
        Some((b[0] as char, b[1] as char))
    } else {
        None
    }
}

/// Map a trait token to a hard gender constraint (`'f'`/`'m'`), if it names one.
fn gender_of(token: &str) -> Option<char> {
    match token {
        "female" | "woman" | "girl" | "f" => Some('f'),
        "male" | "man" | "boy" | "m" => Some('m'),
        _ => None,
    }
}

/// Map a trait token to a hard language/accent constraint (the prefix letter),
/// if it names one.
fn lang_of(token: &str) -> Option<char> {
    match token {
        "american" | "us" | "usa" => Some('a'),
        "british" | "uk" | "english" | "gb" => Some('b'),
        "spanish" | "spain" => Some('e'),
        "french" | "france" => Some('f'),
        "hindi" | "indian" | "india" => Some('h'),
        "italian" | "italy" => Some('i'),
        "japanese" | "japan" => Some('j'),
        "portuguese" | "brazilian" | "brazil" => Some('p'),
        "chinese" | "mandarin" | "china" => Some('z'),
        _ => None,
    }
}

/// Stable FNV-1a hash so trait→voice tiebreaks are reproducible across runs
/// (unlike `DefaultHasher`, whose output is not guaranteed stable).
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Pick one voice id from `candidates`, preferring those not yet in `used`,
/// breaking ties deterministically by `seed`. Returns `None` if empty.
fn pick<'a>(candidates: &[&'a String], used: &HashSet<String>, seed: u64) -> Option<&'a String> {
    if candidates.is_empty() {
        return None;
    }
    let fresh: Vec<&&String> = candidates.iter().filter(|v| !used.contains(**v)).collect();
    let pool = if fresh.is_empty() {
        candidates.iter().collect::<Vec<_>>()
    } else {
        fresh
    };
    let idx = (seed % pool.len() as u64) as usize;
    Some(pool[idx])
}

/// Resolve every character to a concrete voice id.
///
/// - Explicit ids are used verbatim when present in `available` (else a warning
///   and an auto-assignment).
/// - Trait lists filter `available` by the hard constraints derivable from the
///   voice id (gender, language/accent); soft traits (age/timbre) only seed the
///   deterministic tiebreak that keeps each character's voice distinct.
/// - `speakers` (distinct character keys in first-appearance order) catches any
///   speaker that has no cast entry; `narrator` falls back to `narrator_default`.
///
/// Returns the `character_key → voice_id` map and any human-readable warnings.
pub fn resolve_voices(
    cast: &[CastEntry],
    speakers: &[String],
    available: &[String],
    narrator_default: &str,
) -> (HashMap<String, String>, Vec<String>) {
    let avail_set: HashSet<&String> = available.iter().collect();
    let mut mapping: HashMap<String, String> = HashMap::new();
    let mut used: HashSet<String> = HashSet::new();
    let mut warnings: Vec<String> = Vec::new();

    let assign = |mapping: &mut HashMap<String, String>,
                  used: &mut HashSet<String>,
                  k: String,
                  v: String| {
        used.insert(v.clone());
        mapping.insert(k, v);
    };

    for entry in cast {
        let k = key(&entry.name);
        if mapping.contains_key(&k) {
            continue;
        }
        match &entry.spec {
            VoiceSpec::Explicit(id) => {
                if avail_set.contains(id) {
                    assign(&mut mapping, &mut used, k, id.clone());
                } else {
                    warnings.push(format!(
                        "voice '{id}' for {} not found; auto-assigning",
                        entry.name
                    ));
                    let cands: Vec<&String> = available.iter().collect();
                    if let Some(v) = pick(&cands, &used, fnv1a(&k)) {
                        let v = v.clone();
                        assign(&mut mapping, &mut used, k, v);
                    }
                }
            }
            VoiceSpec::Traits(traits) => {
                let want_gender: Option<char> = traits.iter().find_map(|t| gender_of(t));
                let want_lang: Option<char> = traits.iter().find_map(|t| lang_of(t));
                let seed = fnv1a(&format!("{}|{}", entry.name, traits.join(",")));

                // Filter by hard constraints, relaxing language first, then
                // gender, if nothing matches.
                let matches = |vg: Option<(char, char)>, use_lang: bool, use_gender: bool| -> bool {
                    let Some((l, g)) = vg else { return false };
                    (!use_gender || want_gender.is_none_or(|wg| wg == g))
                        && (!use_lang || want_lang.is_none_or(|wl| wl == l))
                };
                let mut chosen: Option<String> = None;
                for (use_lang, use_gender) in [(true, true), (false, true), (false, false)] {
                    let cands: Vec<&String> = available
                        .iter()
                        .filter(|v| matches(lang_gender(v), use_lang, use_gender))
                        .collect();
                    if let Some(v) = pick(&cands, &used, seed) {
                        chosen = Some(v.clone());
                        break;
                    }
                }
                if let Some(v) = chosen {
                    assign(&mut mapping, &mut used, k, v);
                } else {
                    warnings.push(format!("no voice matched traits for {}", entry.name));
                }
            }
        }
    }

    // Any speaker without a cast entry: narrator → default, others auto-assign.
    for sp in speakers {
        if mapping.contains_key(sp) {
            continue;
        }
        if sp == "narrator" {
            mapping.insert(sp.clone(), narrator_default.to_string());
            used.insert(narrator_default.to_string());
            continue;
        }
        let cands: Vec<&String> = available.iter().collect();
        if let Some(v) = pick(&cands, &used, fnv1a(sp)) {
            let v = v.clone();
            warnings.push(format!("no cast entry for '{sp}'; auto-assigned {v}"));
            assign(&mut mapping, &mut used, sp.clone(), v);
        }
    }

    // Guarantee a narrator mapping exists for unattributed narration.
    mapping
        .entry("narrator".to_string())
        .or_insert_with(|| narrator_default.to_string());

    (mapping, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voices() -> Vec<String> {
        // A representative slice of the real catalog (lang/gender by prefix).
        [
            "af_heart", "af_bella", "af_alloy", "am_michael", "am_adam", "bf_emma",
            "bf_alice", "bm_george", "bm_lewis", "if_sara", "im_nicola",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    #[test]
    fn parses_inline_cast_traits_and_explicit() {
        let input = "# Cast\nALICE: female, american, young\nBOB: male, british\nNARRATOR: af_heart\n\n---\n\nALICE: Hello.\n";
        let s = parse(input, None).unwrap();
        assert_eq!(s.cast.len(), 3);
        assert_eq!(
            s.cast[0],
            CastEntry {
                name: "ALICE".into(),
                spec: VoiceSpec::Traits(vec!["female".into(), "american".into(), "young".into()]),
            }
        );
        assert_eq!(s.cast[2].spec, VoiceSpec::Explicit("af_heart".into()));
    }

    #[test]
    fn cast_block_is_not_spoken() {
        let input = "# Cast\nALICE: af_bella\n\n---\n\nALICE: Hi there.\n";
        let s = parse(input, None).unwrap();
        assert_eq!(s.speeches.len(), 1);
        assert_eq!(s.speeches[0].character, "alice");
        assert_eq!(s.speeches[0].text, "Hi there.");
    }

    #[test]
    fn multiline_speech_and_narration_fallback() {
        let input = "Once upon a time.\nThe door creaked.\n\nALICE: Line one.\nStill Alice.\n";
        let s = parse(input, None).unwrap();
        assert_eq!(s.speeches.len(), 2);
        assert_eq!(s.speeches[0].character, "narrator");
        assert_eq!(s.speeches[0].text, "Once upon a time. The door creaked.");
        assert_eq!(s.speeches[1].character, "alice");
        assert_eq!(s.speeches[1].text, "Line one. Still Alice.");
    }

    #[test]
    fn colon_in_prose_is_not_a_speaker() {
        let input = "He said: this is not a speaker line.\n";
        let s = parse(input, None).unwrap();
        assert_eq!(s.speeches.len(), 1);
        assert_eq!(s.speeches[0].character, "narrator");
        assert!(s.speeches[0].text.starts_with("He said: this is not"));
    }

    #[test]
    fn parentheticals_are_stripped() {
        let input = "ALICE: (whispering) come closer (beat) now.\n";
        let s = parse(input, None).unwrap();
        assert_eq!(s.speeches[0].text, "come closer now.");
    }

    #[test]
    fn interruption_cue_sets_next_speech() {
        let input = "ALICE: I really think we should--\nBOB: Stop.\n";
        let s = parse(input, None).unwrap();
        assert_eq!(s.speeches.len(), 2);
        assert!(!s.speeches[0].interrupts_prev);
        assert!(s.speeches[1].interrupts_prev);
        // Trailing `--` normalized to an em-dash, kept for prosody.
        assert!(s.speeches[0].text.ends_with('\u{2014}'));
    }

    #[test]
    fn separate_cast_file_takes_precedence() {
        let body = "# Cast\nALICE: af_alloy\n\n---\n\nALICE: Hi.\n";
        let s = parse(body, Some("ALICE: af_bella\n")).unwrap();
        assert_eq!(s.cast[0].spec, VoiceSpec::Explicit("af_bella".into()));
    }

    #[test]
    fn resolve_filters_by_gender_and_accent() {
        let cast = vec![
            CastEntry { name: "ALICE".into(), spec: VoiceSpec::Traits(vec!["female".into(), "american".into()]) },
            CastEntry { name: "BOB".into(), spec: VoiceSpec::Traits(vec!["male".into(), "british".into()]) },
        ];
        let speakers = vec!["alice".into(), "bob".into()];
        let (m, _w) = resolve_voices(&cast, &speakers, &voices(), "af_heart");
        assert!(m["alice"].starts_with("af_"));
        assert!(m["bob"].starts_with("bm_"));
    }

    #[test]
    fn resolve_keeps_voices_distinct_and_deterministic() {
        let cast = vec![
            CastEntry { name: "A".into(), spec: VoiceSpec::Traits(vec!["female".into(), "american".into()]) },
            CastEntry { name: "B".into(), spec: VoiceSpec::Traits(vec!["female".into(), "american".into()]) },
        ];
        let speakers = vec!["a".into(), "b".into()];
        let (m1, _) = resolve_voices(&cast, &speakers, &voices(), "af_heart");
        let (m2, _) = resolve_voices(&cast, &speakers, &voices(), "af_heart");
        assert_ne!(m1["a"], m1["b"], "distinct voices for distinct characters");
        assert_eq!(m1, m2, "resolution is deterministic across runs");
    }

    #[test]
    fn resolve_explicit_passthrough_and_narrator_default() {
        let cast = vec![CastEntry { name: "X".into(), spec: VoiceSpec::Explicit("bm_george".into()) }];
        let speakers = vec!["x".into(), "narrator".into()];
        let (m, _w) = resolve_voices(&cast, &speakers, &voices(), "af_heart");
        assert_eq!(m["x"], "bm_george");
        assert_eq!(m["narrator"], "af_heart");
    }

    #[test]
    fn resolve_warns_on_unknown_speaker() {
        let (m, w) = resolve_voices(&[], &["ghost".into()], &voices(), "af_heart");
        assert!(m.contains_key("ghost"));
        assert!(w.iter().any(|s| s.contains("ghost")));
    }
}

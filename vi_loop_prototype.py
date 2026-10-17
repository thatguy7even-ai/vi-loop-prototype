#!/usr/bin/env python3
"""VI talk-to-text loop — phone-home prototype (INN-130).

Arm A (text-only): lexical bind-or-quarantine gate run over the operator's
known talk-to-text error corpus. Pre-registered break condition: if the OOV
gate catches < 50% of known error instances, the design is mis-scoped to the
spectacular failure (Chouchay) instead of the common one.

Arm B (audio): full loop mechanics on real speech.
  first pass  = faster-whisper `tiny`  (stands in for the on-device decoder)
  gate        = lexical binding; unbound word -> quarantine + clip timestamps
  phone-home  = ffmpeg-extract the quarantined span -> faster-whisper `base`
                (the Ears-class listener from INN-128) -> splice result back
  lab angle   = every quarantine logged as JSONL: audio in, what was reached
                for, what came back (hallucination-event dataset w/ ground truth)

Run:  ~/workspace/tts-venv/bin/python vi_loop_prototype.py [--audio file]
Arm A always runs. Arm B needs --audio.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- Arm A ---

# (intended, rendered) pairs from the operator's talk-to-text correction record
# plus the Chouchay incident. `rendered` is what the decoder emitted.
KNOWN_ERRORS = [
    ("touché", "Chouchay"),     # the incident: fluent OOV confabulation
    ("now", "No"),              # sentence-initial now -> No
    ("four and five", "4 from"),# dropped conjunction
    ("gronk", "grant"),         # in-vocab substitution
    ("gronk", "Grok"),          # proper-noun miss
    ("archives", "art guys"),   # in-vocab substitution
    ("account", "count"),        # dropped leading syllable
    ("Demis", "dim Z"),         # name -> words
    ("Demis", "dems"),          # name -> near-miss
]

BREAK_CATCH_RATE = 0.50


def load_lexicon(path=os.path.join(HERE, "words.txt")):
    lex = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                lex.add(w)
    return lex


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text):
    return _WORD_RE.findall(text)


def normalize_word(word):
    """Punctuation/possessive normalization BEFORE lexicon lookup.
    Without this, 'cookie,' and 'there.' quarantine as OOV — the gate's
    'word' must be a normalized token, not raw decoder output."""
    w = word.lower().strip()
    w = w.strip(".,!?;:\"()[]{}")
    w = w.strip("'")
    if w.endswith("'s"):
        w = w[:-2]
    elif w.endswith("s'"):
        w = w[:-2]
    return w


def is_bound(word, lex):
    w = normalize_word(word)
    if not w:
        return True
    if w in ("a", "i"):
        return True
    if w.isdigit():
        return True  # numerals are bound by form
    return w in lex


def gate(text, lex):
    """Bind-or-quarantine. Returns (bound_words, quarantined_words)."""
    bound, quar = [], []
    for tok in tokenize(text):
        (bound if is_bound(tok, lex) else quar).append(tok)
    return bound, quar


def arm_a(lex):
    print("=== ARM A — gate coverage over known error corpus ===")
    caught, missed, partial = [], [], []
    for intended, rendered in KNOWN_ERRORS:
        _, quar = gate(rendered, lex)
        toks = tokenize(rendered)
        if not toks:
            missed.append((intended, rendered, "empty"))
        elif len(quar) == len(toks):
            caught.append((intended, rendered))
        elif quar:
            partial.append((intended, rendered, quar))
        else:
            missed.append((intended, rendered, "all-bound"))
    n = len(KNOWN_ERRORS)
    rate = len(caught) / n
    for intended, rendered in caught:
        print(f"  CAUGHT  {rendered!r:12} (intended {intended!r})")
    for intended, rendered, why in missed:
        print(f"  MISSED  {rendered!r:12} (intended {intended!r}) [{why}]")
    for intended, rendered, quar in partial:
        print(f"  PARTIAL {rendered!r:12} (intended {intended!r}) quarantined={quar}")
    print(f"  catch rate: {len(caught)}/{n} = {rate:.0%}")
    if rate < BREAK_CATCH_RATE:
        print(f"  BREAK CONDITION TRIGGERED: {rate:.0%} < {BREAK_CATCH_RATE:.0%} — "
              "the OOV gate is mis-scoped to the spectacular failure.")
    else:
        print("  break condition not triggered.")
    print()
    return {"caught": len(caught), "missed": len(missed),
            "partial": len(partial), "rate": rate,
            "break": rate < BREAK_CATCH_RATE}


# ---------------------------------------------------------------- Arm B ---

def _whisper():
    from faster_whisper import WhisperModel
    return WhisperModel


def first_pass(audio_path, model_name="tiny"):
    WhisperModel = _whisper()
    local_tiny = os.path.join(HERE, "models", "tiny")
    # Local dir (fetched via curl; the venv's vendored httpx2 cannot parse
    # this environment's proxy vars, so hub download is unavailable).
    model_id = local_tiny if model_name == "tiny" and os.path.isdir(local_tiny) else model_name
    model = WhisperModel(model_id, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    out = []
    for s in segments:
        words = []
        for w in (s.words or []):
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
        out.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                    "words": words})
    return info.language, out


def extract_clip(audio_path, start, end, pad=0.4):
    s = max(0.0, start - pad)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.2f}", "-to", f"{end + pad:.2f}",
         "-i", audio_path, "-ac", "1", "-ar", "16000", tmp.name],
        check=True)
    return tmp.name


def phone_home(clip_path, model_name="base"):
    """The Ears-class listener: audio clip in, best-effort true word out."""
    WhisperModel = _whisper()
    # Local snapshot (same proxy-vendoring reason as first_pass).
    local_base = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Systran--faster-whisper-base/"
        "snapshots/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66")
    model_id = local_base if model_name == "base" and os.path.isdir(local_base) else model_name
    model = WhisperModel(model_id, device="cpu", compute_type="int8")
    t0 = time.time()
    segments, _ = model.transcribe(clip_path)
    text = " ".join(s.text.strip() for s in segments)
    dt = time.time() - t0
    return text, dt


def arm_b(audio_path, lex, budget_s=1.0):
    print("=== ARM B — phone-home loop on real audio ===")
    print(f"  audio: {audio_path}")
    lang, segs = first_pass(audio_path)
    print(f"  first-pass (tiny) language: {lang}, segments: {len(segs)}")

    events = []
    for s in segs:
        for w in s["words"]:
            if not is_bound(w["word"], lex):
                events.append(w)
    print(f"  quarantined by lexical gate: {len(events)}")
    for w in events:
        print(f"    {w['word']!r} @ {w['start']:.2f}s–{w['end']:.2f}s")

    log_path = os.path.join(HERE, "hallucination-events.jsonl")
    splices = []
    with open(log_path, "w", encoding="utf-8") as log:
        for w in events:
            clip = extract_clip(audio_path, w["start"], w["end"])
            try:
                heard, dt = phone_home(clip)
            finally:
                os.unlink(clip)
            over = dt > budget_s
            entry = {
                "audio_in": os.path.basename(clip),
                "reached_for": w["word"],
                "came_back": heard,
                "t_start": round(w["start"], 2),
                "t_end": round(w["end"], 2),
                "phone_home_s": round(dt, 3),
                "over_budget": over,
            }
            log.write(json.dumps(entry) + "\n")
            splices.append((w["word"], heard, dt, over))
            flag = " OVER-BUDGET" if over else ""
            print(f"    phone-home {w['word']!r} -> {heard!r} ({dt:.2f}s){flag}")

    print(f"  event log: {log_path} ({len(events)} events)")
    print()
    return {"quarantined": len(events), "splices": splices, "log": log_path}


def main():
    ap = argparse.ArgumentParser(description="VI talk-to-text loop prototype")
    ap.add_argument("--audio", help="audio file for Arm B")
    args = ap.parse_args()

    lex = load_lexicon()
    print(f"lexicon: {len(lex)} entries\n")
    a = arm_a(lex)
    b = arm_b(args.audio, lex) if args.audio else None
    if not args.audio:
        print("(Arm B skipped: no --audio given)")
    print("=== DONE ===")
    return {"arm_a": a, "arm_b": b}


if __name__ == "__main__":
    main()

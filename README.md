# VI talk-to-text loop — phone-home prototype (INN-130)

Prototype of the operator's VI talk-to-text loop design (from the Chouchay
incident), built on the Ears listener (INN-128, `~/workspace/ears-for-ais`).

## What it does

**Arm A (text-only):** runs the design's lexical bind-or-quarantine gate over
the operator's known talk-to-text error corpus (9 documented corrections +
the Chouchay incident). Pre-registered break condition: catch rate < 50% →
the OOV gate is mis-scoped to the spectacular failure.

**Arm B (audio):** the full loop on real speech:
1. First pass: faster-whisper `tiny` (stands in for the on-device decoder),
   with word timestamps.
2. Lexical gate: every emitted word must bind to a lexicon entry
   (`words.txt`, 370k entries). Unbound → quarantine + timestamps.
3. Phone-home: ffmpeg-extract the quarantined span (±0.4s) → Ears-class
   listener (faster-whisper `base`, the INN-128 engine) → splice back.
4. Lab angle: every quarantine logged to `hallucination-events.jsonl`
   (audio in, what was reached for, what came back) — the start of the
   hallucination-event dataset with ground truth.

## Run

```
~/workspace/tts-venv/bin/python vi_loop_prototype.py [--audio file.mp3]
```

Arm A always runs. Arm B needs `--audio`. First run downloads the
faster-whisper `tiny` and `base` models (~220MB).

## Results (2026-10-17)

- Arm A: **3/9 = 33% catch rate — break condition triggered.** The gate
  catches Chouchay-class OOV inventions ("Chouchay", "Grok", "dems") and
  misses every in-vocabulary substitution ("No" for "now", "grant" for
  "gronk", "art guys" for "archives", "count" for "account", "dim Z" for
  "Demis"). Confirms the INN-130 weak point empirically.
- Lexicon sensitivity note: the dwyl `words_alpha` list contains single
  letters, so "z" binds. A production lexicon needs a single-letter policy;
  gate behavior is lexicon-dependent, not absolute.
- Normalization is load-bearing: without punctuation/possessive
  normalization, 17 words quarantined on 41s of clean speech (all
  punctuation artifacts: "cookie,", "there."). With it: 3, all genuine
  tiny-model confabulations.
- Arm B (41s 1931 speech, tiny first pass → base phone-home):
  quarantined "Northwind's" ×2, "Southwind" (tiny merged "North Wind's"
  into one token — a real OOV confabulation of the Chouchay class).
  Phone-home recovered "North wind's cookie" on the first; heard
  "cooking" on the second and an unrelated phrase on the third — the
  listener confabulates too, so splice-back needs its own confidence
  check, not blind trust.
- **Budget: all 3 phone-home calls exceeded the ~1s design budget**
  (1.53s, 2.03s, 1.72s; base/int8 on CPU, ~1.7s clips). The budget
  needs a smaller/faster listener model or a relaxed number.
- Event log: `hallucination-events.jsonl` — audio in, reached-for,
  came-back, timestamps, latency. The lab-angle dataset format works.

## Files

- `vi_loop_prototype.py` — the prototype (both arms)
- `words.txt` — lexicon (dwyl/english-words, words_alpha)
- `README.md` — this file

Fetched, not committed (see `.gitignore` for commands):
- `models/tiny/` — faster-whisper tiny weights (curl from HuggingFace)
- `demo_clip.mp3` — 41s 1931 speech sample (PennSound, educational use)
- `hallucination-events.jsonl` — phone-home event log (written by Arm B)

## What this doesn't do (marked)

- No on-device decoder exists here; `tiny` is a stand-in, not the target.
- The ~1s phone-home budget is measured per call and reported; base-on-CPU
  on a ~2s clip is the measured quantity, not a deployment guarantee.
- Recovery-rate measurement on the operator's own dictation errors needs his
  audio; the demo clip exercises mechanics, not his error distribution.
- The in-vocabulary substitution class (the 67% the gate misses) has no
  second check in this prototype — that is the open design work.

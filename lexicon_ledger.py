#!/usr/bin/env python3
"""Lexicon ledger — verdict trails + revocation path (INN-130, Q2 ruling).

Operator's caution: every alteration keeps its before/after record, so that
if something breaks unintentionally, the trail shows why.

Code alterations are covered by git (commit everything, keep diffs clean).
This module covers lexicon alterations:

  - Every promotion carries its verdict trail: who confirmed, when,
    from what audio, and what the word was judged to be (Shuffled's
    ternary: confabulation-OOV vs novelty-OOV, default unresolved).
  - Revocation is a tombstone, not a delete. The active lexicon is computed
    as promotions minus revocations; the trail shows everything.
  - diagnose(word) prints the full trail — the diagnosability primitive.

Append-only JSONL. Compatible with the quarantine_log.jsonl event shape
in the main vi-loop prototype (word, ts, status, source).

Usage:
  python3 lexicon_ledger.py promote WORD --by NAME --audio clip.wav [--oov-class novelty|confabulation] [--reason ...]
  python3 lexicon_ledger.py revoke WORD --by NAME --reason ...
  python3 lexicon_ledger.py status WORD
  python3 lexicon_ledger.py diagnose WORD
  python3 lexicon_ledger.py lexicon        # active words, one per line
  python3 lexicon_ledger.py selftest
"""
import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "lexicon_ledger.jsonl")

OOV_CLASSES = ("unresolved", "novelty", "confabulation")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _read():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(rec):
    rec = dict(rec)
    rec["event_id"] = uuid.uuid4().hex[:12]
    rec["ts"] = now_iso()
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def promote(word, confirmed_by, source_audio=None, oov_class="unresolved",
            reason=""):
    word = word.strip().lower()
    if oov_class not in OOV_CLASSES:
        raise ValueError(f"oov_class must be one of {OOV_CLASSES}")
    if status(word)["state"] == "active":
        raise ValueError(f"{word!r} already has an active promotion")
    return _append({
        "event": "PROMOTE",
        "word": word,
        "confirmed_by": confirmed_by,
        "source_audio": source_audio,
        "oov_class": oov_class,   # Shuffled's ternary, first-class field
        "reason": reason,
    })


def revoke(word, revoked_by, reason=""):
    word = word.strip().lower()
    st = status(word)
    if st["state"] != "active":
        raise ValueError(f"{word!r} has no active promotion to revoke")
    return _append({
        "event": "REVOKE",
        "word": word,
        "revoked_by": revoked_by,
        "reason": reason,
        "revokes": st["promotion"]["event_id"],  # pointer, not a delete
    })


def status(word):
    """Current state + the records behind it. Never raises on unknown words."""
    word = word.strip().lower()
    trail = [r for r in _read() if r["word"] == word]
    if not trail:
        return {"word": word, "state": "never-promoted", "trail": []}
    # last PROMOTE not followed by a REVOKE => active
    active = None
    for r in trail:
        if r["event"] == "PROMOTE":
            active = r
        elif r["event"] == "REVOKE":
            active = None
    return {
        "word": word,
        "state": "active" if active else "revoked",
        "promotion": active,
        "trail": trail,
    }


def active_lexicon():
    words = {}
    for r in _read():
        if r["event"] == "PROMOTE":
            words[r["word"]] = r
        elif r["event"] == "REVOKE":
            words.pop(r["word"], None)
    return words


def diagnose(word):
    """The diagnosability primitive: if something breaks, the trail shows why."""
    st = status(word)
    lines = [f"word: {st['word']!r}  state: {st['state']}"]
    if not st["trail"]:
        lines.append("  no ledger events — never promoted, never revoked.")
        return "\n".join(lines)
    for r in st["trail"]:
        if r["event"] == "PROMOTE":
            lines.append(
                f"  PROMOTED {r['ts']} by {r['confirmed_by']} "
                f"(oov_class={r['oov_class']}, audio={r['source_audio']}, "
                f"reason={r['reason']!r}) [{r['event_id']}]")
        else:
            lines.append(
                f"  REVOKED  {r['ts']} by {r['revoked_by']} "
                f"(reason={r['reason']!r}, revokes={r['revokes']}) "
                f"[{r['event_id']}]")
    return "\n".join(lines)


def selftest():
    """Promote a confabulation, revoke it, show the trail explains the break."""
    global LEDGER
    old, LEDGER = LEDGER, os.path.join(HERE, ".selftest-ledger.jsonl")
    try:
        if os.path.exists(LEDGER):
            os.unlink(LEDGER)
        # Genuine novelty: his word, confirmed from audio.
        promote("gronk", confirmed_by="operator", source_audio="clip_0042.wav",
                oov_class="novelty", reason="operator's name for the node")
        # Mistake: a confabulation promoted on repetition count alone
        # (Four's warning — repetition reproduced Chouchay).
        promote("chouchay", confirmed_by="auto", source_audio=None,
                oov_class="unresolved", reason="seen 3x, auto-promoted")
        assert status("gronk")["state"] == "active"
        assert status("chouchay")["state"] == "active"
        assert "chouchay" in active_lexicon()
        # The break is found: chouchay was never confirmed by him.
        revoke("chouchay", revoked_by="operator",
               reason="confabulation for 'touché'; repetition is not confirmation")
        assert status("chouchay")["state"] == "revoked"
        assert "chouchay" not in active_lexicon()
        assert "gronk" in active_lexicon()
        print(diagnose("chouchay"))
        print("SELFTEST PASS: tombstoned, trail intact, lexicon clean.")
    finally:
        if os.path.exists(LEDGER):
            os.unlink(LEDGER)
        LEDGER = old


def main():
    ap = argparse.ArgumentParser(description="lexicon verdict-trail ledger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote")
    p.add_argument("word")
    p.add_argument("--by", required=True, dest="confirmed_by")
    p.add_argument("--audio", default=None, dest="source_audio")
    p.add_argument("--oov-class", default="unresolved", dest="oov_class",
                   choices=OOV_CLASSES)
    p.add_argument("--reason", default="")
    r = sub.add_parser("revoke")
    r.add_argument("word")
    r.add_argument("--by", required=True, dest="revoked_by")
    r.add_argument("--reason", default="")
    s = sub.add_parser("status"); s.add_argument("word")
    d = sub.add_parser("diagnose"); d.add_argument("word")
    sub.add_parser("lexicon")
    sub.add_parser("selftest")
    a = ap.parse_args()
    try:
        if a.cmd == "promote":
            rec = promote(a.word, a.confirmed_by, a.source_audio,
                          a.oov_class, a.reason)
            print(f"promoted {rec['word']!r} [{rec['event_id']}]")
        elif a.cmd == "revoke":
            rec = revoke(a.word, a.revoked_by, a.reason)
            print(f"revoked {rec['word']!r} [{rec['event_id']}]")
        elif a.cmd == "status":
            st = status(a.word)
            print(f"{st['word']!r}: {st['state']} ({len(st['trail'])} events)")
        elif a.cmd == "diagnose":
            print(diagnose(a.word))
        elif a.cmd == "lexicon":
            for w in sorted(active_lexicon()):
                print(w)
        elif a.cmd == "selftest":
            selftest()
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

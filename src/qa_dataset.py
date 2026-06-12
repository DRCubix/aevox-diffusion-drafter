#!/usr/bin/env python3
"""QA the distillation dataset: completeness, reasoning structure, degeneration."""
import json, glob, statistics as st, re, collections

rows = []
for f in sorted(glob.glob("/home/nemo1/distill_shards/*.jsonl")):
    for line in open(f):
        rows.append(json.loads(line))
print(f"total examples: {len(rows)}\n")

# 1. source distribution
src = collections.Counter(r["source"] for r in rows)
print("source distribution:", dict(src))

# 2. completion length
lens = [r["n_completion_tokens"] for r in rows]
lens_s = sorted(lens)
def pct(p): return lens_s[min(len(lens_s)-1, int(p*len(lens_s)))]
print(f"\ncompletion tokens: mean={st.mean(lens):.0f} median={pct(.5)} p90={pct(.9)} "
      f"p95={pct(.95)} max={max(lens)}")

# 3. empties / cap-truncated
empty = sum(1 for n in lens if n == 0)
near_cap = sum(1 for n in lens if n >= 8190)  # hit max_tokens=8192 -> truncated
print(f"empty completions: {empty} ({100*empty/len(rows):.1f}%)")
print(f"hit 8192 cap (truncated mid-gen): {near_cap} ({100*near_cap/len(rows):.1f}%)")

# 4. reasoning structure: opened+closed <think>
def has_think(t): return "<think>" in t or "</think>" in t
opened = sum(1 for r in rows if "<think>" in r["completion"])
closed = sum(1 for r in rows if "</think>" in r["completion"])
both = sum(1 for r in rows if "<think>" in r["completion"] and "</think>" in r["completion"])
print(f"\nreasoning tags: has <think>={100*opened/len(rows):.0f}%  has </think>={100*closed/len(rows):.0f}%  "
      f"both (complete think)={100*both/len(rows):.0f}%")
# completions that have a final answer after </think> (non-empty post-think text)
final_ok = 0
for r in rows:
    c = r["completion"]
    if "</think>" in c and len(c.split("</think>")[-1].strip()) > 0:
        final_ok += 1
print(f"has non-empty answer after </think>: {100*final_ok/len(rows):.0f}%")

# 5. degeneration: detect long verbatim repetition in the tail (looping)
def is_degenerate(t):
    t = t[-1200:]
    for w in (40, 80, 160):           # window sizes (chars)
        if len(t) >= 3*w and t[-w:] == t[-2*w:-w] == t[-3*w:-2*w]:
            return True
    return False
degen = sum(1 for r in rows if is_degenerate(r["completion"]))
print(f"\ndegenerate (tail repetition / loop): {degen} ({100*degen/len(rows):.1f}%)")

# 6. id sanity: completion_ids length matches n_completion_tokens, ints
bad_ids = 0
for r in rows[:2000]:
    ci = r["completion_ids"]
    if len(ci) != r["n_completion_tokens"] or (ci and not isinstance(ci[0], int)):
        bad_ids += 1
print(f"id/length mismatches (first 2000): {bad_ids}")

# 7. print 2 short samples (head+tail) per a couple sources for eyeballing
print("\n" + "="*70 + "\nSAMPLES\n" + "="*70)
shown = collections.Counter()
for r in rows:
    if shown[r["source"]] >= 1:
        continue
    shown[r["source"]] += 1
    c = r["completion"]
    print(f"\n--- [{r['source']}] prompt: {r['prompt'][:120]!r}")
    print(f"    completion ({r['n_completion_tokens']} toks) HEAD: {c[:300]!r}")
    print(f"    ...TAIL: {c[-220:]!r}")
    if sum(shown.values()) >= 4:
        break

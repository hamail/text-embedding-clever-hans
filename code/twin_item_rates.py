#!/usr/bin/env python3
"""twin_item_rates.py -- the item-level view of the matched twins.

Closes two reviewer requests in one pass over the stimuli you already embed for
Supplementary Table 1:

  1. How often does the role reversal beat the meaning-preserving rewrite?
     The proportion of items with rho < 1, with a confidence interval, plus the
     median and interquartile range of the paired distance difference, plus the
     same proportion broken down by predicate pair and by sentence frame.

  2. Does the uncertainty survive the structure of the stimuli?
     The same ratio and interval under a CLUSTERED bootstrap that resamples whole
     entity pairs, frames or predicates rather than items, so that a shared
     stimulus component is resampled as one unit.  Report these beside the
     analytic t-interval that Supplementary Table 1 carries.

Nothing here changes an existing number.  It adds the item-level quantities the
ratio alone cannot express.

Usage (from the directory holding chvlm.py and e2_mechanism.py):
  python twin_item_rates.py --models unsup-simcse bge-base qwen3-0.6b
  python twin_item_rates.py --models all --out results/twin_item_rates.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

import chvlm
import e2_mechanism as e2

OUT = "results/twin_item_rates.json"


# ---------------------------------------------------------------------------
# stimulus bookkeeping
# ---------------------------------------------------------------------------

def load_twins() -> tuple[list[dict], list[int], list[int]]:
    """The 400 matched twins, as (rows, kept row indices, reversed row indices).

    Mirrors cmd_minimal_pairs exactly: label 1 is the symmetric predicate, whose
    name transposition keeps the meaning; label 0 is the asymmetric predicate,
    whose identical transposition reverses the roles.
    """
    rows = (list(map(json.loads, open(e2.MINIMAL_PAIRS)))
            if os.path.exists(e2.MINIMAL_PAIRS) else e2.build_minimal_pairs())
    twins: dict[int, dict] = {}
    for k, r in enumerate(rows):
        twins.setdefault(r["match_id"], {})[r["label"]] = k
    ids = sorted(k for k, v in twins.items() if 0 in v and 1 in v)
    return rows, [twins[i][1] for i in ids], [twins[i][0] for i in ids]


def entity_pair(sentence: str) -> str:
    """The unordered name pair of a sentence, as the clustering unit."""
    found = sorted({n for n in e2.NAMES if f" {n} " in f" {sentence} "
                    or sentence.startswith(n + " ")})
    return "|".join(found) if found else "?"


def _frame_patterns() -> list[tuple[str, "re.Pattern"]]:
    """One regex per frame.  Matching on the literal prefix fails for a frame whose
    fixed text contains a slot, such as "During {year}, ...", so every slot becomes
    a wildcard and the surrounding punctuation does the identifying."""
    import re
    pats = []
    for f in e2.FRAMES:
        rx = re.escape(f)
        for slot, sub in (("\\{year\\}", r"\d{4}"), ("\\{city\\}", r"[A-Z][a-z]+"),
                          ("\\{S\\}", r".+?")):
            rx = rx.replace(slot, sub)
        pats.append((f, re.compile("^" + rx + "$")))
    return pats


_FRAMES = None


def frame_of(sentence: str) -> str:
    """Which of the five frames produced this sentence."""
    global _FRAMES
    if _FRAMES is None:
        _FRAMES = _frame_patterns()
    for f, rx in _FRAMES:
        if rx.match(sentence):
            return f
    return "?"


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------

def boot_proportion(flag: np.ndarray, groups: np.ndarray | None,
                    reps: int = 2000, seed: int = 0, level: float = 0.90):
    """Percentile interval for mean(flag).

    groups=None resamples items.  Otherwise whole groups are drawn with
    replacement and every item of a drawn group is taken, so a shared stimulus
    component moves as one unit.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        n = len(flag)
        vals = [flag[rng.integers(0, n, n)].mean() for _ in range(reps)]
    else:
        keys = np.array(sorted(set(groups.tolist())))
        index = {k: np.flatnonzero(groups == k) for k in keys}
        vals = []
        for _ in range(reps):
            drawn = keys[rng.integers(0, len(keys), len(keys))]
            vals.append(flag[np.concatenate([index[k] for k in drawn])].mean())
    a = (1 - level) / 2
    return float(np.percentile(vals, 100 * a)), float(np.percentile(vals, 100 * (1 - a)))


MIN_CLUSTERS = 30     # below this a cluster bootstrap is not trustworthy


def boot_ratio(logr: np.ndarray, groups: np.ndarray,
               reps: int = 2000, seed: int = 0, level: float = 0.90):
    """Clustered percentile interval for the geometric-mean ratio exp(mean(log rho)).

    Returns None when there are too few clusters.  A cluster bootstrap resamples
    the clusters, so with a handful of them the interval is driven by which of the
    few were drawn and comes out narrower than the item interval, which reads as
    reassurance and is an artefact.  The 5 frames and 8 predicates fall in that
    range; the entity pairs do not.
    """
    keys = np.array(sorted(set(groups.tolist())))
    if len(keys) < MIN_CLUSTERS:
        return None
    rng = np.random.default_rng(seed)
    index = {k: np.flatnonzero(groups == k) for k in keys}
    vals = []
    for _ in range(reps):
        drawn = keys[rng.integers(0, len(keys), len(keys))]
        vals.append(np.exp(logr[np.concatenate([index[k] for k in drawn])].mean()))
    a = (1 - level) / 2
    return float(np.percentile(vals, 100 * a)), float(np.percentile(vals, 100 * (1 - a)))


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--precision", default=None)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    models = list(chvlm.MODELS) if a.models == ["all"] else a.models

    rows, kept, rev = load_twins()
    pair = np.array([entity_pair(rows[k]["s1"]) for k in kept])
    pred = np.array([rows[k]["predicate"] for k in rev])        # the asymmetric one
    frame = np.array([frame_of(rows[k]["s1"]) for k in kept])
    print(f"[twins] {len(kept)} items | {len(set(pair))} entity pairs | "
          f"{len(set(pred))} asymmetric predicates | {len(set(frame))} frames\n")

    out = {}
    hdr = (f"{'model':16s} {'rev closer':>11s} {'90% CI (item)':>16s} "
           f"{'90% CI (pair)':>16s} {'median dd':>10s} {'IQR':>17s} {'ratio':>7s} "
           f"{'ratio CI (pair)':>17s}")
    print(hdr)
    for name in models:
        spec = chvlm.at_precision(name, a.precision)
        m = chvlm.load(spec, eager=False)
        E1 = chvlm.embed(m, [r["s1"] for r in rows], bs=128)
        E2 = chvlm.embed(m, [r["s2"] for r in rows], bs=128)
        del m
        mu = np.vstack([E1, E2]).mean(0)                        # as in cmd_minimal_pairs
        d = 1 - chvlm.cosine(E1, E2, mu)

        d_kept, d_rev = d[kept], d[rev]
        closer = (d_rev < d_kept).astype(float)                 # reversal nearer
        diff = d_kept - d_rev                                   # >0 when reversal nearer
        logr = np.log(d_rev / d_kept)

        p = float(closer.mean())
        ci_i = boot_proportion(closer, None, a.reps)
        ci_p = boot_proportion(closer, pair, a.reps)
        q1, med, q3 = (float(x) for x in np.percentile(diff, [25, 50, 75]))
        ratio = float(np.exp(logr.mean()))
        rci_p = boot_ratio(logr, pair, a.reps)            # the only unit with enough
        rci_f = boot_ratio(logr, frame, a.reps)           # clusters; the other two
        rci_d = boot_ratio(logr, pred, a.reps)            # return None by design

        by = {}
        for label, key in (("predicate", pred), ("frame", frame)):
            g = {}
            for k in sorted(set(key.tolist())):
                sel = key == k
                g[k] = dict(n=int(sel.sum()), closer=float(closer[sel].mean()),
                            ratio=float(np.exp(logr[sel].mean())))
            by[label] = g

        out[name] = dict(n=len(kept), reversal_closer=p, ci_item=ci_i, ci_entity_pair=ci_p,
                         median_diff=med, iqr=[q1, q3], ratio=ratio,
                         ratio_ci_entity_pair=rci_p, ratio_ci_frame=rci_f,
                         ratio_ci_predicate=rci_d,
                         tost=e2.tost_log_ratio(logr), by=by)
        rp = f"{rci_p[0]:7.3f}-{rci_p[1]:<9.3f}" if rci_p else f"{'n/a':>17s}"
        print(f"{name:16s} {p:11.3f} {ci_i[0]:7.3f}-{ci_i[1]:<8.3f} "
              f"{ci_p[0]:7.3f}-{ci_p[1]:<8.3f} {med:10.4f} "
              f"{q1:8.4f}-{q3:<8.4f} {ratio:7.3f} {rp}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n[done] wrote {a.out}")
    print("Read the two intervals together: if the entity-pair interval is wider than the")
    print("item interval and still excludes the value you are arguing against, the result")
    print("survives the clustering.  The per-predicate and per-frame breakdowns say whether")
    print("the effect rests on a few templates or holds across all of them.")


if __name__ == "__main__":
    main()

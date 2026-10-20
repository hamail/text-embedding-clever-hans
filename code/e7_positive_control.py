#!/usr/bin/env python3
"""e7_positive_control.py -- does anything get these items right?

Why this exists
---------------
Every system measured in the paper so far fails on the same stimuli, which leaves one
objection open: that the items themselves are the problem rather than the bi-encoders.
A positive control closes it. If a scorer that reads the two sentences together
separates a role reversal from a meaning-preserving rewrite on the same 400 items,
then the items are sound and the failure is a property of the bi-encoder design. If
nothing separates them, that is worth knowing before submission rather than after.

It also anchors the equivalence bound. The 1.25 margin is a bioequivalence
convention; a cross-encoder's ratio on the same items says what a scorer that does use
the roles actually achieves, which makes the bound interpretable rather than arbitrary.

The measure
-----------
A cross-encoder emits a score per pair, not an embedding, so the distance analogue is
  d = 1 - score            (score in [0, 1]; for NLI, score = P(entailment))
and the ratio is the same quantity the paper reports for the bi-encoders,
  rho = d(reversed) / d(kept),
aggregated in the log domain and tested with the same TOST at the same margin, by
calling e2_mechanism.tost_log_ratio directly so the statistics cannot drift.

  rho near 1   the scorer cannot separate them -- role-blind, like the bi-encoders
  rho >> 1     the scorer puts the reversal much further -- the items are sound

Both stimulus sets are run: the matched twins, which the central claim rests on, and
the passive contrast.

Usage (from ~/CHVLM/revised)
----------------------------
  python e7_positive_control.py                      # the two defaults below
  python e7_positive_control.py --models cross-encoder/stsb-roberta-base
  python e7_positive_control.py --nli cross-encoder/nli-deberta-v3-base
  python e7_positive_control.py --limit 50           # smoke test first
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import chvlm
import e2_mechanism as e2

OUT = "results/e7"

# The STS cross-encoder is the one already used in the paper as the overlap-excess
# reference, so its ratio is directly comparable with numbers already reported. The
# NLI model is the stronger test: a role reversal should read as contradiction, and
# P(entailment) should collapse.
DEFAULT_STS = ["cross-encoder/stsb-roberta-base"]
DEFAULT_NLI = ["cross-encoder/nli-deberta-v3-base"]


def score_pairs(name: str, pairs: list[tuple[str, str]], kind: str, bs: int) -> np.ndarray:
    """Similarity in [0, 1] for each pair. For NLI this is P(entailment)."""
    from sentence_transformers import CrossEncoder
    import torch

    model = CrossEncoder(name, max_length=128)
    raw = model.predict(pairs, batch_size=bs, show_progress_bar=True,
                        convert_to_numpy=True)
    if kind == "nli":
        # (contradiction, entailment, neutral) for the deberta NLI cross-encoders;
        # take the entailment column by the model's own label map rather than by
        # position, because the orderings differ between checkpoints.
        labels = {v.lower(): k for k, v in model.config.id2label.items()}
        col = labels.get("entailment")
        if col is None:
            raise SystemExit(f"{name}: no 'entailment' label in {model.config.id2label}")
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        s = (e / e.sum(axis=1, keepdims=True))[:, col]
    else:
        s = np.asarray(raw, dtype=float)
        if s.ndim > 1:
            raise SystemExit(f"{name}: expected one score per pair, got shape {s.shape}")
        # stsb cross-encoders emit [0, 1]; clip rather than rescale, and say so if the
        # range looks like it is not that scale at all.
        if s.min() < -0.1 or s.max() > 1.1:
            print(f"  note: {name} scores span {s.min():.2f} to {s.max():.2f}; "
                  f"min-max rescaling to [0, 1] for the distance")
            s = (s - s.min()) / (s.max() - s.min())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.clip(s, 0.0, 1.0)


def ratios(d_kept: np.ndarray, d_rev: np.ndarray) -> np.ndarray:
    """Per-item log ratio, with the floor the paper uses for identical embeddings."""
    eps = 1e-9
    return np.log(np.maximum(d_rev, eps) / np.maximum(d_kept, eps))


def report(tag: str, name: str, d_kept: np.ndarray, d_rev: np.ndarray) -> dict:
    res = e2.tost_log_ratio(ratios(d_kept, d_rev))
    print(f"{name[:34]:34s} {d_kept.mean():8.4f} {d_rev.mean():11.4f} "
          f"{res['ratio']:9.3f} ({res['ci90'][0]:.3f}-{res['ci90'][1]:.3f})"
          f" {res['p_not_further']:12.1e} {res['p_tost']:9.1e}  "
          f"{'yes' if res['role_blind'] else 'NO':>3s}")
    return dict(set=tag, model=name, d_kept=float(d_kept.mean()),
                d_reversed=float(d_rev.mean()), **res)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=DEFAULT_STS,
                    help="STS cross-encoders, scored as a similarity in [0, 1]")
    ap.add_argument("--nli", nargs="*", default=DEFAULT_NLI,
                    help="NLI cross-encoders, scored as P(entailment)")
    ap.add_argument("--n-passive", type=int, default=400)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N items of each set, for a smoke test")
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()

    # ---- stimulus set 1: the matched twins, which the central claim rests on.
    # Two rows share a match_id: label 1 keeps the meaning under the name swap,
    # label 0 reverses the roles. Both are a pair, so a cross-encoder scores them
    # directly without any change to the design.
    rows = (list(map(json.loads, open(e2.MINIMAL_PAIRS)))
            if os.path.exists(e2.MINIMAL_PAIRS) else e2.build_minimal_pairs())
    twins: dict[int, dict] = {}
    for r in rows:
        twins.setdefault(r["match_id"], {})[r["label"]] = r
    ids = sorted(k for k, v in twins.items() if 0 in v and 1 in v)
    if a.limit:
        ids = ids[:a.limit]
    kept_pairs = [(twins[i][1]["s1"], twins[i][1]["s2"]) for i in ids]
    rev_pairs = [(twins[i][0]["s1"], twins[i][0]["s2"]) for i in ids]

    # ---- stimulus set 2: the passive contrast, both conditions against the same base
    pit = e2.passive_items(a.n_passive, a.seed)
    if a.limit:
        pit = pit[:a.limit]
    p_kept = [(it["base"], it["kept"]) for it in pit]
    p_rev = [(it["base"], it["reversed"]) for it in pit]

    print(f"matched twins: {len(ids)} items | passive contrast: {len(pit)} items")
    print(f"margin {e2.MARGIN}, the bound fixed before any model was run\n")

    out = []
    for kind, names in (("sts", a.models), ("nli", a.nli)):
        for name in names:
            print(f"--- {name}  ({'P(entailment)' if kind == 'nli' else 'similarity'}) ---")
            print(f"{'set':34s} {'d kept':>8s} {'d reversed':>11s} "
                  f"{'ratio (90% CI)':>24s} {'P not further':>12s} {'P alike':>9s}  blind")
            for tag, kp, rp in (("matched twins", kept_pairs, rev_pairs),
                                ("passive contrast", p_kept, p_rev)):
                s_kept = score_pairs(name, kp, kind, a.bs)
                s_rev = score_pairs(name, rp, kind, a.bs)
                out.append(report(tag, f"{tag}", 1 - s_kept, 1 - s_rev) | dict(model=name))
            print()

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "positive_control.json")
    json.dump(dict(args=vars(a), results=out), open(path, "w"), indent=2)
    print(f"[done] wrote {path}")
    print("\nHow to read this: 'blind = yes' means the scorer fails the same way the")
    print("bi-encoders do. 'blind = NO' with a ratio well above 1 is the positive")
    print("control succeeding -- the items are sound and the failure is the encoder's.")


if __name__ == "__main__":
    main()

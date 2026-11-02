#!/usr/bin/env python3
"""Generate the per-predicate supplementary table from twin_item_rates.json.

The aggregate ratio averages over eight predicate pairs that behave very
differently.  Reporting that spread pre-empts the obvious objection, and it
carries the argument that the variation is not role sensitivity: a role-sensitive
score would exceed one on every pair, since the asymmetric transposition changes
the meaning where the symmetric one does not.

Usage (from ~/CHVLM/revised, after twin_item_rates.py):
  python make_stab_predicates.py                       # three models, as in the text
  python make_stab_predicates.py --models all          # every model in the JSON
  python make_stab_predicates.py --out results/tables/T22_body.tex
"""

from __future__ import annotations

import argparse
import json

# The eight pairs of e2_mechanism.PREDICATES, symmetric member first.  The
# symmetric predicate leaves the name transposition meaning-preserving; the
# asymmetric one makes the identical transposition a role reversal.
PAIRS = [("met", "defeated"), ("married", "hired"), ("divorced", "sued"),
         ("fought", "trained"), ("worked with", "worked for"),
         ("argued with", "reported to"), ("played against", "voted for"),
         ("competed against", "wrote to")]

NICE = {"unsup-simcse": "unsup-SimCSE", "sup-simcse": "sup-SimCSE",
        "bge-base": "BGE-base", "e5-base": "E5-base", "gte-base": "GTE-base",
        "mpnet-base": "MPNet-base", "mxbai-large": "mxbai-large",
        "retrained": "retrained", "qwen3-0.6b": "Qwen3-0.6B",
        "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B"}

CAPTION = (
    r"\textbf{Ordering a role reversal depends on the predicate rather than on the "
    r"model.} For each of the eight pairs, the distance ratio $\rho$ and the share of "
    r"that pair's items on which the role reversal is the closer of the two. The "
    r"symmetric predicate leaves the name transposition meaning-preserving; the "
    r"asymmetric one makes the identical transposition a role reversal, therefore a "
    r"role-sensitive score would put $\rho$ above one on every row. None does. The "
    r"spread runs in both directions: some pairs are separated decisively and well "
    r"above the margin, while on others the reversal is the closer of the two in every "
    r"item. No model is therefore role-sensitive across predicates, and which case a "
    r"given relation falls into cannot be read off the model. The \emph{worked with} "
    r"and \emph{worked for} pair differs by one preposition, so it holds lexical "
    r"content almost exactly, and the three models agree there.")


def build(data: dict, models: list[str]) -> str:
    cols = "l" + "rr" * len(models)
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{NICE.get(m, m)}}}" for m in models)
    sub = " & ".join(r"$\rho$ & closer" for _ in models)
    spans = " ".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(models)))
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             f"predicate pair & {head} \\\\", spans,
             f" & {sub} \\\\", r"\midrule"]
    for sym, asym in PAIRS:
        cells = []
        for m in models:
            v = data[m]["by"]["predicate"].get(asym)
            cells.append("-- & --" if v is None
                         else f"${v['ratio']:.3f}$ & ${v['closer']*100:.0f}\\%$")
        lines.append(f"\\emph{{{sym}}} / \\emph{{{asym}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule"]
    tally = []
    for m in models:
        p = data[m]["by"]["predicate"]
        n = sum(1 for _, a in PAIRS if a in p and p[a]["ratio"] > 1)
        tally.append(f"\\multicolumn{{2}}{{c}}{{{n} of 8}}")
    lines.append(r"rows with $\rho>1$ & " + " & ".join(tally) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/twin_item_rates.json")
    ap.add_argument("--models", nargs="+",
                    default=["unsup-simcse", "bge-base", "qwen3-0.6b"])
    ap.add_argument("--out", default="results/tables/T22_body.tex")
    ap.add_argument("--stable", action="store_true",
                    help="wrap in \\stable{stab:predicates}{caption}{...} for the supplement")
    a = ap.parse_args()

    data = json.load(open(a.json))
    models = list(data) if a.models == ["all"] else a.models
    missing = [m for m in models if m not in data]
    if missing:
        raise SystemExit(f"not in {a.json}: {', '.join(missing)}")

    body = build(data, models)
    text = (f"% T22  Per-predicate ratios  (generated, do not edit)\n{body}\n")
    if a.stable:
        text = ("\\stable{stab:predicates}{" + CAPTION + "}{%\n" + body + "}\n")
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(text)
    print(text)
    print(f"[done] wrote {a.out}")


if __name__ == "__main__":
    main()

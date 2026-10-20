#!/usr/bin/env python3
"""Emit T19_body.tex, the positive control, from results/e7/positive_control.json.

tables.py stops at T18, so this one table was never generated, although
fig2_mock.py reads it for panel d and the supplement prints it as Table 21.
Every value here comes from the JSON; nothing is typed in.

    cd ~/CHVLM/revised && python make_T19.py
"""

from __future__ import annotations
import argparse, json, os

SETS = ["matched twins", "passive contrast"]          # row order, as in the supplement


def prec(x: float) -> str:
    """Three decimals below ten, two below a hundred, one above: the precision
    the supplement already uses for 0.901, 44.74 and 427.8."""
    return f"{x:.3f}" if x < 10 else (f"{x:.2f}" if x < 100 else f"{x:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/e7/positive_control.json")
    ap.add_argument("--out", default="results/tables/T19_body.tex")
    a = ap.parse_args()

    d = json.load(open(a.json))
    nli = set(d["args"]["nli"])
    rows = d["results"]

    # one row per (stimulus set, scorer), similarity scorer first within each set
    order = {s: i for i, s in enumerate(SETS)}
    rows.sort(key=lambda r: (order.get(r["set"], 99), r["model"] in nli))

    out = [r"\begin{tabular}{lllrrrr}", r"\toprule",
           "stimulus set & scorer & trained on & score kept & score reversed & "
           r"ratio & $90\%$ CI \\", r"\midrule"]
    for r in rows:
        name = r["model"].split("/")[-1]
        trained = "entailment" if r["model"] in nli else "similarity"
        lo, hi = r["ci90"]
        out.append(
            f"{r['set']} & \\texttt{{{name}}} & {trained} & "
            f"${1 - r['d_kept']:.3f}$ & ${1 - r['d_reversed']:.3f}$ & "
            f"${prec(r['ratio'])}$ & ${prec(lo)}$--${prec(hi)}$ \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    body = "\n".join(out)
    with open(a.out, "w") as f:
        f.write("% T19  A positive control: what a scorer that reads both "
                "sentences achieves  (generated, do not edit)\n" + body + "\n")
    print(body)
    print(f"\n[done] {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()

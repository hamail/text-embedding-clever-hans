#!/usr/bin/env python3
"""Mock-up of a four-panel Figure 2.

Panels a and b are the current figure, unchanged.  Panels c and d are new, and both
are drawn from the generated table bodies rather than from invented numbers, so the
mock-up can be read as a draft of the real thing and not only as a layout sketch.

  c  the scale trend against precision   (Supplementary Table 2 / T01b)
  d  the positive control                (Supplementary Table 21 / T19)

The logic the four panels make, left to right and top to bottom: one worked instance,
then every model against a bound fixed in advance, then the confound ruled out, then
what a scorer that is not blind looks like on the same items.

Run:  python fig2_mock.py --tables /mnt/user-data/outputs/tables --out .
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- style, unchanged
BLUE = "#1B6CA8"      # the meaning-preserving rewrite / the matched twins
ORANGE = "#C55A11"    # the role reversal / the passive contrast
GREEN = "#2E8B57"
INK = "#1A1A1A"
INK2 = "#555555"
GRID = "#D8D8D8"
NEUTRAL = "#8A8A8A"
BAND = "#EAF3EC"

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#B0B0B0", "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


# ---------------------------------------------------------------- reading the tables
def _clean(cell: str) -> str:
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    cell = cell.replace(r"\%", "%").replace(r"\_", "_").replace(r"\&", "&")
    cell = re.sub(r"\\[a-zA-Z]+", "", cell)
    cell = cell.replace("$", "").replace("{", "").replace("}", "")
    return cell.replace("--", "-").strip()


def load(tabdir, tid):
    path = os.path.join(tabdir, f"{tid}_body.tex")
    head, rows = None, []
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s.startswith("%") or "&" not in s:
            continue
        s = re.sub(r"\\\\.*$", "", s)
        cells = [_clean(c) for c in s.split("&")]
        if head is None:
            head = cells
        else:
            rows.append(cells)
    return head, rows


def colof(head, key):
    for i, h in enumerate(head):
        if h.strip().lower() == key.strip().lower():
            return i
    for i, h in enumerate(head):
        if key.lower() in h.lower():
            return i
    raise SystemExit(f"no column matching {key!r} in {head}")


def num(cell):
    s = cell.replace("%", "").strip()
    if s in ("-", "--", "", "nan"):
        return None
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " ", s)
    m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m[0]) if m else None


def rangeof(cell):
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " ", cell.replace("%", "").strip())
    m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return (float(m[0]), float(m[1])) if len(m) >= 2 else None


def wrap(s, n=58):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return lines


# =========================================================================== panel a
def panel_a(ax, tabdir, n_items, dist_table):
    head, rows = load(tabdir, "T18")
    jm, jd = colof(head, "model"), colof(head, "domain")
    jo, jp = colof(head, "original"), colof(head, "same meaning")
    jw = colof(head, "opposite meaning")
    jk, jr, jt = colof(head, "d kept"), colof(head, "d reversed"), colof(head, "ratio")

    items = [r for r in rows if r[jm] == "BGE-base"]
    items.sort(key=lambda r: num(r[jt]))
    total, show = len(items), items[:n_items]

    ax.axis("off")
    ax.text(-0.01, 1.005, "a", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, va="bottom")
    y = 0.985
    for r in show:
        ratio, dk, dr = num(r[jt]), num(r[jk]), num(r[jr])
        rev_closer = dr < dk
        ax.text(0.0, y, " ".join(r[jd].upper()), fontsize=7.6, color=INK2)
        ax.text(1.0, y, f"ratio {ratio:.2f}", fontsize=8, color=INK2, ha="right")
        y -= 0.035
        for ln in wrap(r[jo]):
            ax.text(0.0, y, ln, fontsize=9.2, color=INK)
            y -= 0.030
        y -= 0.006
        for txt, d, col, closer in ((r[jp], dk, BLUE, not rev_closer),
                                    (r[jw], dr, ORANGE, rev_closer)):
            ax.text(0.0, y, f"{d:.4f}", fontsize=8.6, color=col,
                    fontweight="bold" if closer else "normal")
            if closer:
                ax.plot([0.115], [y + 0.008], marker="o", ms=5, color=col,
                        clip_on=False)
            for ln in wrap(txt, 52):
                ax.text(0.145, y, ln, fontsize=9.0, color=col)
                y -= 0.030
            y -= 0.004
        ax.plot([0.0, 1.0], [y + 0.012, y + 0.012], color="#E4E4E4", lw=0.8,
                clip_on=False)
        y -= 0.024
    ax.set_xlim(0, 1)
    ax.set_ylim(y, 1.0)
    ax.text(0.0, y + 0.004,
            f"Distances under BGE-base; 1 - centred cosine. A filled dot marks the "
            f"closer of the two.\nAll {total} items, every model, in Supplementary "
            f"Table {dist_table}.", fontsize=7.6, color=INK2, va="top")


# =========================================================================== panel b
def panel_b(ax, tabdir):
    h1, r1 = load(tabdir, "T02")
    jm, jr, jci = colof(h1, "model"), colof(h1, "ratio"), colof(h1, "CI")
    twins = {r[jm]: (num(r[jr]), rangeof(r[jci])) for r in r1}

    h2, r2 = load(tabdir, "T01")
    jm2, jr2, jc2 = colof(h2, "model"), colof(h2, "ratio"), colof(h2, "CI")
    passive = {r[jm2]: (num(r[jr2]), rangeof(r[jc2])) for r in r2}

    models = sorted(((m, twins[m][0], twins[m][1]) for m in twins), key=lambda t: t[1])

    ax.text(-0.10, 1.005, "b", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, va="bottom")
    for i, (name, ratio, ci) in enumerate(models):
        if name in passive:
            pr, pci = passive[name]
            if pci:
                ax.plot(pci, [i + 0.22, i + 0.22], color=ORANGE, lw=1.4, alpha=0.45,
                        solid_capstyle="butt")
            ax.plot([pr], [i + 0.22], marker="s", ms=4.2, color=ORANGE,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=2)
        if ci:
            ax.plot(ci, [i, i], color=BLUE, lw=1.8, alpha=0.6, solid_capstyle="butt")
        ax.plot([ratio], [i], marker="o", ms=5.5, color=BLUE,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(1.46, i, f"{ratio:.2f}", fontsize=8.6, color=BLUE, va="center")
    ax.axvspan(0.80, 1.25, color=BAND, zorder=0)
    ax.axvline(1.0, color=INK, ls=(0, (4, 3)), lw=1.2)
    for mk, col, lab in (("o", BLUE, "matched twins"),
                         ("s", ORANGE, "passive contrast")):
        ax.plot([], [], mk, color=col, ls="", label=lab)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m[0] for m in models], fontsize=8.8, color=INK)
    ax.invert_yaxis()
    ax.set_ylim(len(models) - 0.3, -1.1)
    ax.text(1.025, -0.85, "equivalence band", fontsize=8.2, color=GREEN, ha="center")
    ax.annotate("", xy=(0.18, len(models) - 0.62), xytext=(0.46, len(models) - 0.62),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    ax.text(0.49, len(models) - 0.62, "reversal closer", fontsize=8.2,
            color=INK2, va="center")
    ax.set_xlim(0, 1.44)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00, 1.25])
    ax.set_xlabel("distance to the reversal $\\div$ distance to the paraphrase")
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# ====================================================== panel c, the scale confound
def panel_precision(ax, tabdir):  # panel d
    """int8 is the one precision measured at all three sizes, so it carries the line.

    The other precisions hang off each size as small open markers.  Drawn this way the
    reader sees a long climb with short stubs on it, which is the claim: size moves the
    ratio by 0.424 and precision by at most 0.033.
    """
    head, rows = load(tabdir, "T01b")
    jm, jp = colof(head, "model"), colof(head, "precision")
    jpr, jtr = colof(head, "passive ratio"), colof(head, "twin ratio")

    SIZE = {"Qwen3-0.6B": 0.6, "Qwen3-4B": 4.0, "Qwen3-8B": 8.0}
    data = {}
    for r in rows:
        data.setdefault(r[jm], {})[r[jp]] = (num(r[jtr]), num(r[jpr]))

    ax.text(-0.13, 1.005, "d", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, va="bottom")
    ax.axhspan(0.80, 1.25, color=BAND, zorder=0)
    ax.axhline(1.0, color=INK, ls=(0, (4, 3)), lw=1.2, zorder=1)

    # Direct labels rather than a legend: panel b already carries the key for the whole
    # figure, and a second legend in here collided with the orange line.
    for k, (col, mk, lab) in enumerate(((BLUE, "o", "matched twins"),
                                        (ORANGE, "s", "passive contrast"))):
        xs = [SIZE[m] for m in SIZE if "int8" in data.get(m, {})]
        ys = [data[m]["int8"][k] for m in SIZE if "int8" in data.get(m, {})]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs, ys = [xs[i] for i in order], [ys[i] for i in order]
        ax.plot(xs, ys, "-", color=col, lw=1.8, zorder=3)
        ax.plot(xs, ys, mk, color=col, ms=7, markeredgecolor="white",
                markeredgewidth=0.9, zorder=4)
        # Labels sit in clear space beside each line, not on it and not past its end,
        # where they were clipped by the axes.
        ax.text(*((1.15, 0.822) if k == 0 else (1.15, 0.487)), lab, fontsize=8.6,
                color=col, va="center")
        # the other precisions, offset left of their size so they do not sit on the line
        for m, prs in data.items():
            for p, v in prs.items():
                if p == "int8":
                    continue
                ax.plot([SIZE[m] * 0.78], [v[k]], mk, ms=5.4, color="white",
                        markeredgecolor=col, markeredgewidth=1.3, zorder=4)
                ax.plot([SIZE[m] * 0.78, SIZE[m]], [v[k], prs["int8"][k]],
                        color=col, lw=0.9, alpha=0.55, zorder=2)

    # Both callouts are quantities of the passive-contrast series and are drawn on it,
    # in its colour.  A first version put the 0.424 arrow on the twins line, which spans
    # 0.284; the registered figure of 0.424 is the passive climb from 0.465 to 0.889.
    ax.annotate("", xy=(9.3, 0.465), xytext=(9.3, 0.889),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.3))
    ax.text(9.65, 0.677, "size\n0.424", fontsize=8.6, color=ORANGE, va="center")
    # The precision gap is marked at 4B and labelled in the clear space below the
    # orange line, with a leader.  Every position closer to the markers collided with
    # one of them or sat on a line.
    ax.annotate("", xy=(4.0, 0.600), xytext=(4.0, 0.637),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.1))
    ax.plot([4.15, 4.9], [0.606, 0.545], color=INK2, lw=0.7, alpha=0.8)
    ax.text(4.95, 0.538, "precision\n0.033", fontsize=8.0, color=ORANGE,
            ha="left", va="center")
    ax.text(0.43, 1.245, "open markers: fp16 and fp32, which coincide at 0.6B",
            fontsize=7.8, color=INK2, va="top")

    ax.set_xscale("log")
    ax.set_xticks([0.6, 4, 8])
    ax.set_xticklabels(["0.6B", "4B", "8B"])
    ax.set_xlim(0.40, 13.5)
    ax.set_ylim(0.38, 1.30)
    ax.set_xlabel("parameters")
    ax.set_ylabel("ratio")
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)


# ==================================================== panel d, the positive control
def panel_control(ax, tabdir):  # panel c
    """What a scorer that does use the roles reaches, on the very same items.

    Every point in b falls inside the band, which leaves open the objection that the
    band is simply too wide to fail.  The entailment cross-encoder on the matched twins
    is the one measurement in the paper that lands outside it.
    """
    head, rows = load(tabdir, "T19")
    js = colof(head, "stimulus set")
    jt, jr, jci = colof(head, "trained on"), colof(head, "ratio"), colof(head, "CI")

    h1, r1 = load(tabdir, "T02")
    tw = [num(r[colof(h1, "ratio")]) for r in r1]
    h2, r2 = load(tabdir, "T01")
    pa = [num(r[colof(h2, "ratio")]) for r in r2]

    # The passive-contrast controls reach 44.7 and 427.8.  Plotting those forces a log
    # axis on which the whole 0.7 to 1.5 region, which is the region the claim lives in,
    # collapses to a few pixels.  The twins carry the central claim, so the panel is the
    # twins on a linear axis and the passive numbers are stated underneath.
    ax.text(-0.13, 1.005, "c", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, va="bottom")
    ax.axvspan(0.80, 1.25, color=BAND, zorder=0)
    ax.axvline(1.0, color=INK, ls=(0, (4, 3)), lw=1.2, zorder=1)

    ax.plot([min(tw), max(tw)], [0, 0], color=BLUE, lw=6, alpha=0.28,
            solid_capstyle="round", zorder=2)
    ax.plot(tw, [0] * len(tw), "o", color=BLUE, ms=4.6, markeredgecolor="white",
            markeredgewidth=0.6, zorder=3)
    ax.text(max(tw) + 0.045, 0, f"{min(tw):.3f} to {max(tw):.3f}", fontsize=8.3,
            color=BLUE, va="center")

    y, labels = 1, ["eleven bi-encoders"]
    for r in rows:
        if r[js] != "matched twins":
            continue
        v, ci = num(r[jr]), rangeof(r[jci])
        blind = v < 1.25
        col = BLUE if blind else GREEN
        mk = "o" if blind else "D"          # shape as well as colour, for tritanopia
        if ci:
            ax.plot(ci, [y, y], color=col, lw=2.0, alpha=0.6, solid_capstyle="butt",
                    zorder=3)
        ax.plot([v], [y], marker=mk, ms=7.5, color=col, markeredgecolor="white",
                markeredgewidth=0.9, zorder=4)
        right = max(v, ci[1] if ci else v)
        ax.text(right + 0.035, y, f"{v:.3f}", fontsize=8.6, color=col, va="center",
                fontweight="bold" if not blind else "normal")
        if not blind:
            # Stated on the row itself; an arrow from below crossed the axis label, and
            # the note about reading both sentences sat on top of the 0.901 marker.
            ax.text(right + 0.265, y, "outside the band", fontsize=8.2, color=GREEN,
                    va="center")
        labels.append(f"cross-encoder\ntrained on {r[jt]}")
        y += 1

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.3, color=INK)
    ax.invert_yaxis()
    ax.set_ylim(len(labels) - 0.35, -0.7)
    ax.set_xlim(0.62, 2.10)
    ax.set_xticks([0.75, 1.00, 1.25, 1.50])
    ax.set_xlabel("ratio, matched twins")
    ax.xaxis.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(1.025, -0.52, "equivalence band", fontsize=8.0, color=GREEN, ha="center")
    pr = [num(r[jr]) for r in rows if r[js] == "passive contrast"]
    ax.text(0.5, -0.195,
            f"On the passive contrast the same two scorers reach {min(pr):.0f} and "
            f"{max(pr):.0f}, against {min(pa):.3f} to {max(pa):.3f} for the eleven "
            f"bi-encoders.",
            transform=ax.transAxes, fontsize=7.8, color=INK2, ha="center")


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="/mnt/user-data/outputs/tables")
    ap.add_argument("--out", default=".")
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--dist-table", type=int, default=20)
    a = ap.parse_args()

    fig = plt.figure(figsize=(13.0, 11.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.30, 1.0], height_ratios=[2.05, 1.0],
                          wspace=0.34, hspace=0.30)
    panel_a(fig.add_subplot(gs[0, 0]), a.tables, a.items, a.dist_table)
    panel_b(fig.add_subplot(gs[0, 1]), a.tables)
    panel_control(fig.add_subplot(gs[1, 0]), a.tables)
    panel_precision(fig.add_subplot(gs[1, 1]), a.tables)

    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"fig2_mock.{ext}"), facecolor="white")
    plt.close(fig)
    print("wrote fig2_mock.png and fig2_mock.pdf")


if __name__ == "__main__":
    main()

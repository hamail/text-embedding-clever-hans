#!/usr/bin/env python3
"""Rebuild all five main-text figures directly from the generated tables.

Why this script exists
----------------------
The previous figures were drawn from figdata.json, which could hold demo values,
and three of them drifted from the tables: Figure 2a's six worked items did not
match the distance table at all (the negation item even had the wrong sentence
closer), Figure 4c reported a single pooled 1.6x where the text reports ten
matched cells, and every panel carried a DEMO DATA watermark.

This script reads the SAME table bodies the manuscript sets, so a figure cannot
disagree with a table unless the table itself changed.  There is no watermark and
no demo path.

Input
-----
A directory of the generated LaTeX table bodies, one file per table:

    T01_body.tex  T04_body.tex  T09_body.tex  T10_body.tex  T13_body.tex
    T14_body.tex  T15_body.tex  T16_body.tex  T17_body.tex  T18_body.tex

Usage
-----
    python make_figures.py --tables results/tables --out figs
    python make_figures.py --tables results/tables --out figs --only 2 4

Every figure is written as both .png (300 dpi, for LaTeX) and .pdf (vector).

Two numbers in Figure 1b are benchmark scores for BGE-base that are not in any
generated table.  Set them with --stsb and --top1, or the panel is drawn with the
values marked as unverified.  Everything else comes from the tables.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# --------------------------------------------------------------------------
# Style.  Three categorical colours in a fixed order, never cycled, checked for
# colour-vision separation (worst adjacent pair dE 8.8 protan, 22.3 normal).
# Grey is a neutral for reference lines only and is never a series.
# --------------------------------------------------------------------------
BLUE = "#1B6CA8"     # the meaning-preserving rewrite / encoders
ORANGE = "#C55A11"   # the role reversal / decoders
GREEN = "#2E8B57"    # third series where one is needed
INK = "#1A1A1A"
INK2 = "#555555"
GRID = "#D8D8D8"
NEUTRAL = "#8A8A8A"  # chance, random level, reference lines

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#B0B0B0",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 4.5,
})


# --------------------------------------------------------------------------
# Reading the tables
# --------------------------------------------------------------------------
def _clean(cell: str) -> str:
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    cell = cell.replace(r"\%", "%").replace(r"\_", "_").replace(r"\&", "&")
    cell = re.sub(r"\\[a-zA-Z]+", "", cell)
    cell = cell.replace("$", "").replace("{", "").replace("}", "")
    return cell.replace("--", "-").strip()


def load(tabdir: str, tid: str) -> tuple[list[str], list[list[str]]]:
    """Header and rows of one generated table body."""
    path = os.path.join(tabdir, f"{tid}_body.tex")
    if not os.path.exists(path):
        hits = [f for f in os.listdir(tabdir) if f.startswith(tid + "_")]
        if not hits:
            raise SystemExit(f"missing table {tid} in {tabdir}")
        path = os.path.join(tabdir, sorted(hits)[0])
    head, rows = None, []
    for line in open(path):
        s = line.strip()
        if s.startswith("%") or "&" not in s:
            continue
        s = re.sub(r"\\\\.*$", "", s)
        cells = [_clean(c) for c in s.split("&")]
        if head is None:
            head = cells
        else:
            rows.append(cells)
    if head is None:
        raise SystemExit(f"no rows parsed from {path}")
    return head, rows


def colof(head: list[str], key: str) -> int:
    """Index of the column matching key.

    Exact matches are tried first.  Substring matching alone is a trap: the language
    column "de" is a substring of the header "model", so a PAWS-X panel built on
    substring lookup silently plotted the model names as AUC values.
    """
    for i, h in enumerate(head):
        if h.strip().lower() == key.strip().lower():
            return i
    for i, h in enumerate(head):
        if key.lower() in h.lower():
            return i
    raise SystemExit(f"no column matching {key!r} in {head}")


def num(cell: str) -> float | None:
    """First number in a cell.  A dash between digits separates a range."""
    s = cell.replace("%", "").strip()
    if s in ("-", "--", "", "nan"):
        return None
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " ", s)
    m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m[0]) if m else None


def rangeof(cell: str) -> tuple[float, float] | None:
    """Both ends of a range cell such as 0.423-0.462."""
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " ", cell.replace("%", "").strip())
    m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return (float(m[0]), float(m[1])) if len(m) >= 2 else None


def finish(fig, out: str, name: str) -> None:
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.png and {name}.pdf")


# ==========================================================================
# Figure 1.  The argument, on one item.
# ==========================================================================
def fig1(tabdir, out, stsb=None, top1=None):
    """Three panels: the construction, what evaluation reports, what retrieval returns.

    The two distances come from the distance table (BGE-base, ritonavir item), so
    panel c cannot disagree with Supplementary Table 16.  The two benchmark scores
    in panel b are not in any generated table and must be supplied.
    """
    head, rows = load(tabdir, "T18")
    jm, jo = colof(head, "model"), colof(head, "original")
    jk, jr = colof(head, "d kept"), colof(head, "d reversed")
    jp, jw = colof(head, "same meaning"), colof(head, "opposite meaning")

    hit = [r for r in rows
           if r[jm] == "BGE-base" and r[jo].lower().startswith("ritonavir")]
    if not hit:
        raise SystemExit("the ritonavir item is not in T18 under BGE-base")
    r = hit[0]
    original, para, rev = r[jo], r[jp], r[jw]
    d_kept, d_rev = num(r[jk]), num(r[jr])
    # The table stores distance as 1 - centred cosine, so the cosine is 1 - d.
    cos_kept, cos_rev = 1 - d_kept, 1 - d_rev

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.1, 1.0], hspace=0.30, wspace=0.12)

    def panel(ax, letter, title, edge):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.92, transform=ax.transAxes,
                               facecolor="white", edgecolor=edge, linewidth=1.4,
                               zorder=0))
        ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=INK, va="bottom")
        ax.text(0.05, 1.06, title, transform=ax.transAxes, fontsize=10,
                color=INK2, va="bottom")

    def wrap(s, n=42):
        words, lines, cur = s.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > n:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        return "\n".join(lines)

    # a -- the construction
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", "What the similarity is built on", "#C9C9C9")
    ax.text(0.07, 0.86, "Two sentences, one vocabulary", fontsize=9, color=INK2)
    ax.text(0.07, 0.70, wrap(original), fontsize=9.5, color=INK, va="top")
    ax.plot([0.07, 0.93], [0.50, 0.50], color="#DDDDDD", lw=0.8)
    ax.text(0.07, 0.42, wrap(rev), fontsize=9.5, color=ORANGE, va="top")
    ax.text(0.07, 0.12, "Same words, opposite action.", fontsize=9,
            color=ORANGE, style="italic")

    # b -- what evaluation reports
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", "What evaluation reports", GREEN)
    ax.text(0.07, 0.86, "Benchmark agreement", fontsize=9, color=INK2)
    unver = []
    s_txt = f"{stsb:.2f}" if stsb is not None else "--"
    t_txt = f"{top1:.3f}" if top1 is not None else "--"
    if stsb is None:
        unver.append("STS-B")
    if top1 is None:
        unver.append("top-1")
    ax.text(0.07, 0.68, "STS-B", fontsize=10, color=INK)
    ax.text(0.90, 0.68, s_txt, fontsize=11, color=GREEN, fontweight="bold", ha="right")
    ax.text(0.07, 0.52, "Paraphrase top-1", fontsize=10, color=INK)
    ax.text(0.90, 0.52, t_txt, fontsize=11, color=GREEN, fontweight="bold", ha="right")
    ax.plot([0.07, 0.93], [0.38, 0.38], color="#DDDDDD", lw=0.8)
    ax.text(0.07, 0.24, "Nothing is out of place.", fontsize=9,
            color=GREEN, style="italic")
    if unver:
        ax.text(0.07, 0.10, "pass --stsb / --top1 to fill these",
                fontsize=7.5, color="#B00020")

    # c -- what retrieval returns
    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c", "What retrieval returns", ORANGE)
    ax.text(0.07, 0.86, "Query: the sentence in a", fontsize=9, color=INK2)
    ax.text(0.07, 0.72, "1", fontsize=10, color=ORANGE, fontweight="bold")
    ax.text(0.13, 0.72, wrap(rev, 34), fontsize=9, color=ORANGE, va="top")
    ax.text(0.95, 0.72, f"{cos_rev:.3f}", fontsize=9.5, color=ORANGE, ha="right")
    ax.text(0.07, 0.46, "2", fontsize=10, color=BLUE, fontweight="bold")
    ax.text(0.13, 0.46, wrap(para, 34), fontsize=9, color=BLUE, va="top")
    ax.text(0.95, 0.46, f"{cos_kept:.3f}", fontsize=9.5, color=BLUE, ha="right")
    ax.plot([0.07, 0.93], [0.26, 0.26], color="#DDDDDD", lw=0.8)
    ax.text(0.07, 0.12, "The reversal outranks the faithful rewording.",
            fontsize=8.5, color=ORANGE, style="italic")

    for x0, x1 in ((0.345, 0.365), (0.655, 0.675)):
        fig.add_artist(FancyArrowPatch((x0, 0.62), (x1, 0.62),
                                       transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=11,
                                       color=INK2, lw=1.1))

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.add_patch(Rectangle((0.005, 0.30), 0.99, 0.62, transform=ax.transAxes,
                           facecolor="#FDF3EC", edgecolor=ORANGE, linewidth=1.2))
    ax.text(0.025, 0.74, "The reversed mechanism is returned as the answer.",
            fontsize=11.5, fontweight="bold", color=ORANGE, va="center")
    ax.text(0.025, 0.47,
            "Across 11 models in four families, a sentence that reverses who acts "
            "on whom is ranked above one that preserves the meaning in other words.",
            fontsize=9, color=INK, va="center")
    ax.text(0.005, 0.10,
            "Distances are 1 - centred cosine under BGE-base; the cosines shown are "
            "1 - d. The same item in every model is in Fig. 2.",
            fontsize=7.6, color=INK2, va="center")

    finish(fig, out, "fig1_argument")


# ==========================================================================
# Figure 2.  The finding: worked items, and the ratio in every model.
# ==========================================================================
def fig2(tabdir, out, n_items=6):
    """a: worked items under BGE-base, read from the distance table.
    b: the distance ratio for all eleven models, read from the ratio table.

    Panel a previously carried placeholder values that disagreed with every row of
    the distance table.  Here the items ARE the table rows, sorted by ratio, so the
    filled dot always marks whichever sentence the table says is closer.
    """
    head, rows = load(tabdir, "T18")
    jm, jd = colof(head, "model"), colof(head, "domain")
    jo, jp = colof(head, "original"), colof(head, "same meaning")
    jw = colof(head, "opposite meaning")
    jk, jr, jt = colof(head, "d kept"), colof(head, "d reversed"), colof(head, "ratio")

    items = [r for r in rows if r[jm] == "BGE-base"]
    items.sort(key=lambda r: num(r[jt]))
    total = len(items)
    show = items[:n_items]

    h1, r1 = load(tabdir, "T01")
    jmm, jrr = colof(h1, "model"), colof(h1, "ratio")
    jci = colof(h1, "CI")
    models = [(r[jmm], num(r[jrr]), rangeof(r[jci])) for r in r1]
    models.sort(key=lambda t: t[1])

    fig = plt.figure(figsize=(13.0, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.34)

    # ---- a
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(-0.01, 1.005, "a", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, va="bottom")

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
        for i, (txt, d, col, closer) in enumerate((
                (r[jp], dk, BLUE, not rev_closer),
                (r[jw], dr, ORANGE, rev_closer))):
            lines = wrap(txt, 52)
            ax.text(0.0, y, f"{d:.4f}", fontsize=8.6, color=col,
                    fontweight="bold" if closer else "normal")
            if closer:
                ax.plot([0.115], [y + 0.008], marker="o", ms=5, color=col,
                        clip_on=False)
            for j, ln in enumerate(lines):
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
            f"Table 16.", fontsize=7.6, color=INK2, va="top")

    # ---- b
    ax = fig.add_subplot(gs[0, 1])
    ax.text(-0.10, 1.005, "b", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, va="bottom")
    ys = range(len(models))
    for i, (name, ratio, ci) in enumerate(models):
        col = ORANGE if name.lower().startswith("qwen") else BLUE
        if ci:
            ax.plot(ci, [i, i], color=col, lw=1.6, alpha=0.55,
                    solid_capstyle="butt")
        ax.plot([ratio], [i], marker="o", ms=5.5, color=col,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(1.055, i, f"{ratio:.2f}", fontsize=8.6, color=col, va="center")
    ax.axvspan(0, 1, color="#F6F2EE", zorder=0)
    ax.axvline(1.0, color=INK, ls=(0, (4, 3)), lw=1.2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([m[0] for m in models], fontsize=8.8, color=INK)
    ax.invert_yaxis()
    ax.set_ylim(len(models) - 0.3, -1.1)
    # "treats both alike" above the dashed line, the direction cue below the points,
    # so neither sits on the axis or on a model label.
    ax.text(0.99, -0.85, "treats both alike", fontsize=8.2, color=INK2, ha="right")
    ax.annotate("", xy=(0.22, len(models) - 0.65),
                xytext=(0.54, len(models) - 0.65),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    ax.text(0.57, len(models) - 0.65, "reversal closer", fontsize=8.2,
            color=INK2, va="center")
    ax.set_xlim(0, 1.03)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel("distance to the reversal $\\div$ distance to the paraphrase")
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    finish(fig, out, "fig2_finding")


# ==========================================================================
# Figure 3.  The dissociation, layer by layer.
# ==========================================================================
def fig3(tabdir, out):
    """Every layer in the table is plotted, starting at layer 1.

    The previous version began at layer 5, so the minimum the Results quote
    (0.010 at layer 2) was off the axis.
    """
    head, rows = load(tabdir, "T04")
    jm, jl = colof(head, "model"), colof(head, "layer")
    ja, jp = colof(head, "probe active"), colof(head, "probe passive")
    jr = colof(head, "distance ratio")

    names = []
    for r in rows:
        if r[jm] not in names:
            names.append(r[jm])
    if len(names) != 2:
        print(f"  note: T04 holds {len(names)} models, plotting the first two")
    names = names[:2]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.2),
                             gridspec_kw=dict(hspace=0.34, wspace=0.18))
    for k, name in enumerate(names):
        sub = sorted([r for r in rows if r[jm] == name], key=lambda r: int(num(r[jl])))
        L = [int(num(r[jl])) for r in sub]

        ax = axes[0, k]
        ax.plot(L, [num(r[ja]) for r in sub], "-o", color=BLUE, label="active voice")
        ax.plot(L, [num(r[jp]) for r in sub], "-s", color=GREEN, label="passive voice")
        ax.axhline(50, color=NEUTRAL, ls="--", lw=1.0)
        ax.text(L[0], 52, "chance", fontsize=7.6, color=NEUTRAL, va="bottom")
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_title(f"{name}   ({len(L)} layers)", color=INK)
        ax.yaxis.grid(True, alpha=0.7)
        ax.set_axisbelow(True)
        if k == 0:
            ax.set_ylabel("role read off by a\nlinear probe")
            ax.legend(loc="lower left", ncol=2)

        ax = axes[1, k]
        ax.plot(L, [num(r[jr]) for r in sub], "-o", color=ORANGE)
        ax.axhline(1.0, color=INK, ls=(0, (4, 3)), lw=1.2)
        ax.text(L[0], 1.03, "a similarity that tracked meaning would sit above this line",
                fontsize=8.0, color=INK, va="bottom")
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.set_xlabel("layer")
        ax.yaxis.grid(True, alpha=0.7)
        ax.set_axisbelow(True)
        if k == 0:
            ax.set_ylabel("distance to reversal $\\div$\ndistance to paraphrase")

    axes[0, 0].text(-0.16, 1.16, "a", transform=axes[0, 0].transAxes, fontsize=13,
                    fontweight="bold", color=INK)
    axes[0, 0].text(-0.07, 1.16, "The representation records the roles",
                    transform=axes[0, 0].transAxes, fontsize=10.5,
                    fontweight="bold", color=BLUE)
    axes[1, 0].text(-0.16, 1.16, "b", transform=axes[1, 0].transAxes, fontsize=13,
                    fontweight="bold", color=INK)
    axes[1, 0].text(-0.07, 1.16, "The similarity built on it does not use them",
                    transform=axes[1, 0].transAxes, fontsize=10.5,
                    fontweight="bold", color=ORANGE)

    finish(fig, out, "fig3_dissociation")


# ==========================================================================
# Figure 4.  The mechanism.
# ==========================================================================
def fig4(tabdir, out):
    """c now shows the ten matched cells the text reports, not a pooled average.

    The old panel collapsed every rank and lambda into one bar pair and printed a
    single 1.6x, which is a different statistic from the per-cell ratio the Results
    quote.  Here each cell is a point and the reported range is the axis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1),
                            gridspec_kw=dict(wspace=0.30))

    # ---- a  role directions
    head, rows = load(tabdir, "T09")
    jm = colof(head, "model")
    cols = [("own name", BLUE, "with its own entity"),
            ("other names", "#7FB3D5", "with other entities"),
            ("role vs role", ORANGE, "with other role directions"),
            ("random level", NEUTRAL, "random")]
    ax = axes[0]
    n, w = len(rows), 0.19
    for s, (key, col, lab) in enumerate(cols):
        j = colof(head, key)
        ax.bar([i + (s - 1.5) * w for i in range(n)],
               [num(r[j]) for r in rows], width=w * 0.92, color=col, label=lab,
               edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels([r[jm] for r in rows], rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("alignment (|cosine|)")
    ax.set_title("A role rides on its entity", color=INK)
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)

    # ---- b  rank growth
    head, rows = load(tabdir, "T13")
    sizes = [int(num(h)) for h in head[1:]]
    ax = axes[1]
    series = [r for r in rows if "generic" not in r[0].lower()]
    for i, r in enumerate(series):
        ax.plot(sizes, [num(c) for c in r[1:]], "-o",
                color=[BLUE, GREEN, ORANGE][i % 3], label=r[0])
    ax.axhline(1, color=NEUTRAL, ls="--", lw=1.2)
    ax.text(sizes[0], 2.5, "a single role axis would need 1, at any size",
            fontsize=8.0, color=INK2, va="bottom")
    ax.set_xticks(sizes)
    ax.set_xlabel("entities in the vocabulary")
    ax.set_ylabel("directions for 90%\nof the binding term")
    ax.set_title("So the rank climbs", color=INK)
    ax.legend(loc="upper left", fontsize=8)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)

    # ---- c  the ten matched cells
    head, rows = load(tabdir, "T10")
    jm, je = colof(head, "model"), colof(head, "edit")
    jrk, jl = colof(head, "rank"), colof(head, "lambda")
    jrem, jlost = colof(head, "similarity removed"), colof(head, "STS-B lost")

    per = {}
    for r in rows:
        rem, lost = num(r[jrem]), num(r[jlost])
        if rem:
            per[(r[jm], r[je], r[jrk], r[jl])] = lost / rem
    cells = []
    for (m, ed, rk, lam), v in per.items():
        if "entity" not in ed.lower():
            continue
        ctrl = next((k for k in per
                     if k[0] == m and "control" in k[1].lower()
                     and k[2] == rk and k[3] == lam), None)
        if ctrl:
            cells.append((m, rk, lam, v / per[ctrl]))
    cells.sort(key=lambda t: t[3])
    if not cells:
        raise SystemExit("no matched cells found in T10")
    lo, hi = cells[0][3], cells[-1][3]

    ax = axes[2]
    mods = []
    for c in cells:
        if c[0] not in mods:
            mods.append(c[0])
    for i, (m, rk, lam, ratio) in enumerate(cells):
        col = [BLUE, ORANGE, GREEN][mods.index(m) % 3]
        ax.plot([ratio], [i], marker="o", ms=6, color=col,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(ratio + 0.06, i, f"rank {rk}, $\\lambda$ {lam}", fontsize=7.4,
                color=INK2, va="center")
    ax.axvline(1.0, color=INK, ls=(0, (4, 3)), lw=1.2)
    ax.text(0.98, -1.35, "equal cost", fontsize=8, color=INK2, ha="right")
    for i, m in enumerate(mods):
        ax.plot([], [], marker="o", ls="", color=[BLUE, ORANGE, GREEN][i % 3],
                label=m)
    ax.legend(loc="upper left", fontsize=8, handletextpad=0.4)
    ax.set_yticks([])
    ax.set_ylim(-1.9, len(cells) + 0.4)
    ax.set_xlim(0, max(3.6, hi + 0.7))
    ax.set_xlabel("STS-B lost per unit of swap similarity removed,\n"
                  "role swaps $\\div$ non-role swaps")
    ax.set_title(f"and costs {lo:.1f}-{hi:.1f}$\\times$ more to undo", color=INK)
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    print(f"  Figure 4c: {len(cells)} matched cells, ratio {lo:.4f} to {hi:.4f}"
          f"  -> the prose must say {lo:.1f} to {hi:.1f}")

    for k, letter in enumerate("abc"):
        axes[k].text(-0.14, 1.10, letter, transform=axes[k].transAxes,
                     fontsize=13, fontweight="bold", color=INK)

    finish(fig, out, "fig4_mechanism")


# ==========================================================================
# Figure 5.  The registered battery.
# ==========================================================================
def fig5(tabdir, out, protocol="01337a7", analysis="341a4b1"):
    from matplotlib.patches import Patch

    def head_block(ax, letter, title, verdict, verdict_col, registered):
        """Panel letter, title, verdict and the registered criterion, stacked so
        none of the four can land on top of another."""
        ax.text(-0.15, 1.30, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=INK, va="top")
        ax.text(0.0, 1.30, title, transform=ax.transAxes, fontsize=10.5,
                color=INK, va="top")
        ax.text(1.0, 1.30, verdict, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color=verdict_col, ha="right", va="top")
        ax.text(0.0, 1.17, registered, transform=ax.transAxes, fontsize=8,
                color=INK2, va="top")

    fig = plt.figure(figsize=(12.8, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.26,
                          top=0.855, bottom=0.07, left=0.08, right=0.97)

    # ---- a  H1 HANS
    head, rows = load(tabdir, "T14")
    jm, jo = colof(head, "model"), colof(head, "overall")
    jci = colof(head, "CI")
    jp = colof(head, "passive")
    js = colof(head, "swap")
    ax = fig.add_subplot(gs[0, 0])
    order = sorted(rows, key=lambda r: num(r[jo]))
    for i, r in enumerate(order):
        ci = rangeof(r[jci])
        if ci:
            ax.plot(ci, [i, i], color=INK, lw=1.4, solid_capstyle="butt")
        ax.plot([num(r[jo])], [i], "o", ms=6, color=INK, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.plot([num(r[jp])], [i], "^", ms=6, color=BLUE, zorder=3)
        ax.plot([num(r[js])], [i], "v", ms=6, color=ORANGE, zorder=3)
    ax.axvspan(0.40, 0.60, color="#EAF3EC", zorder=0)
    ax.axvline(0.5, color=NEUTRAL, ls="--", lw=1.1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([r[jm] for r in order], fontsize=8.8)
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.4, -0.9)
    ax.text(0.50, -0.75, "predicted band", fontsize=8, color=GREEN, ha="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC: true entailment above non-entailment")
    head_block(ax, "a", "H1  HANS entailment", "DID NOT HOLD", ORANGE,
               "registered: every English encoder between 0.40 and 0.60 AUC")
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for mk, col, lab in (("o", INK, "overall"), ("^", BLUE, "non-entailment by passive"),
                         ("v", ORANGE, "by subject/object swap")):
        ax.plot([], [], mk, color=col, label=lab, ls="")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=8)

    # ---- b  H2 PAWS-X
    head, rows = load(tabdir, "T15")
    # Language columns are every column but the model name and the summary column.
    jlang = [i for i, h in enumerate(head)
             if i > 0 and len(h.strip()) <= 3 and "-" not in h]
    langs = [head[i] for i in jlang]
    ax = fig.add_subplot(gs[0, 1])
    tops = []
    for i, r in enumerate(rows):
        vals = [num(r[i2]) for i2 in jlang]
        col = [BLUE, GREEN, ORANGE][i % 3]
        ax.plot(range(len(langs)), vals, "-o", color=col, label=r[0])
        ax.axhline(vals[0] + 0.05, color=col, ls=":", lw=1.0, alpha=0.8)
        tops.append(vals[0] + 0.05)
    ax.text(len(langs) - 1, max(tops) + 0.004, "bound: English + 0.05",
            fontsize=8, color=INK2, ha="right", va="bottom")
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs)
    ax.set_xlabel("PAWS-X language")
    ax.set_ylabel("AUC")
    head_block(ax, "b", "H2  not an English artefact", "HELD", GREEN,
               "registered: no language more than 0.05 AUC above English")
    ax.legend(loc="lower left", fontsize=8)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)

    # ---- c  H3 the excess
    head, rows = load(tabdir, "T16")
    sets = [h for h in head[1:] if h.lower() not in ("positive",)]
    ax = fig.add_subplot(gs[1, 0])
    series = [r for r in rows if "reference" not in r[0].lower()]
    w = 0.8 / len(series)
    for i, r in enumerate(series):
        graded = "ckpt" in r[0].lower()
        ax.bar([j + (i - len(series) / 2) * w for j in range(len(sets))],
               [num(r[colof(head, s)]) for s in sets], width=w * 0.9,
               color=(ORANGE if graded else BLUE), edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(sets, rotation=20, ha="right")
    ax.set_ylabel("overlap excess above the\ncross-encoder")
    head_block(ax, "c", "H3  the excess generalises", "HELD", GREEN,
               "registered: encoders positive on >= 5/6; graded below base on >= 5/6")
    # Patch proxies, not empty bars: ax.bar([], []) makes an empty container and the
    # legend then draws whatever it can find, which rendered both keys the same blue.
    ax.legend(handles=[Patch(facecolor=BLUE, label="contrastive encoders"),
                       Patch(facecolor=ORANGE, label="graded")],
              loc="upper right", fontsize=8)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)

    # ---- d  H4 transfer
    head, rows = load(tabdir, "T17")
    ax = fig.add_subplot(gs[1, 1])
    hb, rb = load(tabdir, "T14")
    base = min(num(r[colof(hb, "overall")]) for r in rb)
    ax.plot([0], [base], "o", ms=7, mfc="none", color=NEUTRAL)
    ax.plot([-0.16, 0.16], [base, base], color=NEUTRAL, lw=2)
    for i, r in enumerate(rows):
        vals = [num(c) for c in r[1:] if num(c) is not None]
        col = ORANGE if "entity" in r[0].lower() else BLUE
        ax.plot([i + 1] * len(vals), vals, "o", ms=7, mfc="none", color=col)
        ax.plot([i + 1 - 0.16, i + 1 + 0.16], [sum(vals) / len(vals)] * 2,
                color=col, lw=2)
    allv = [num(c) for r in rows for c in r[1:] if num(c) is not None]
    ax.axhspan(max(min(num(c) for c in rows[1][1:] if num(c) is not None),
                   min(num(c) for c in rows[0][1:] if num(c) is not None)),
               min(max(num(c) for c in rows[1][1:] if num(c) is not None),
                   max(num(c) for c in rows[0][1:] if num(c) is not None)),
               color="#FDF3EC", zorder=0)
    ax.text(2.35, max(allv) - 0.02, "the ranges overlap", fontsize=8.5, color=ORANGE,
            ha="right")
    ax.set_xticks(range(len(rows) + 1))
    ax.set_xticklabels(["(base)"] + [r[0] for r in rows], rotation=18, ha="right")
    ax.set_ylabel("HANS AUC after training")
    head_block(ax, "d", "H4  transfer to HANS", "DID NOT HOLD", ORANGE,
               "registered: entity swap above base at every seed, matched control not")
    ax.text(0.02, 0.96, "one point per seed", transform=ax.transAxes, fontsize=8,
            color=INK2)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)

    held = 2
    fig.text(0.5, 0.975,
             f"Registered before the data was touched  ·  protocol {protocol}, "
             f"analysis {analysis}  ·  {held} of 4 held",
             fontsize=9.5, color=INK2, ha="center")

    finish(fig, out, "fig5_battery")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default="results/tables",
                    help="directory holding T01_body.tex and the rest")
    ap.add_argument("--out", default="figs", help="where to write the figures")
    ap.add_argument("--only", nargs="*", type=int, choices=[1, 2, 3, 4, 5],
                    help="build only these figures")
    ap.add_argument("--items", type=int, default=6,
                    help="how many worked items to show in Figure 2a")
    ap.add_argument("--stsb", type=float,
                    help="Figure 1b: STS-B for the model used in the worked item")
    ap.add_argument("--top1", type=float,
                    help="Figure 1b: top-1 paraphrase retrieval for that model")
    ap.add_argument("--protocol", default="01337a7")
    ap.add_argument("--analysis", default="341a4b1")
    a = ap.parse_args()

    if not os.path.isdir(a.tables):
        sys.exit(f"no such directory: {a.tables}")
    want = a.only or [1, 2, 3, 4, 5]

    print(f"reading tables from {a.tables}")
    if 1 in want:
        print("Figure 1"); fig1(a.tables, a.out, a.stsb, a.top1)
    if 2 in want:
        print("Figure 2"); fig2(a.tables, a.out, a.items)
    if 3 in want:
        print("Figure 3"); fig3(a.tables, a.out)
    if 4 in want:
        print("Figure 4"); fig4(a.tables, a.out)
    if 5 in want:
        print("Figure 5"); fig5(a.tables, a.out, a.protocol, a.analysis)
    print(f"\ndone -- {a.out}/ now holds the figures the manuscript expects")


if __name__ == "__main__":
    main()

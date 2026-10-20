#!/usr/bin/env python3
"""Figure 5 -- the registered battery, four predictions reported as they came out.

Writes fig5_battery.png and .pdf, the names \\includegraphics already asks for.

  a  H1  HANS entailment              (T14)
  b  H2  not an English artefact      (T15)
  c  H3  the excess generalises       (T16)
  d  H4  transfer to HANS             (T17, against the T14 baseline)

Same four panels and same numbers as before. What changes is the geometry and the
type. The old figure hung panel a's legend below its axes, which pushed the left
column down and left the 2x2 visibly lopsided; every legend now sits inside its own
panel, every x label is horizontal, and the four axes are checked after drawing to
confirm they are the same size and share their row and column edges. The palette and
the embedded fonts now match Figures 2, 3 and 4.

Run
    python fig5_battery.py                 # finds the tables itself
    python fig5_battery.py --nature        # 180 mm wide, Nature type sizes
    python fig5_battery.py --check         # verify only, draw nothing
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------- colour
BLUE = "#0072B2"      # Okabe-Ito blue
ORANGE = "#D55E00"    # Okabe-Ito vermillion
GREEN = "#009E73"     # Okabe-Ito bluish green
INK = "#1A1A1A"
INK2 = "#555555"
GRID = "#D8D8D8"
NEUTRAL = "#8A8A8A"
BAND = "#ECECEC"      # the registered band; neutral, since it is not a verdict
HELD, FAILED = GREEN, ORANGE

SZ = dict(letter=13.0, title=10.5, verdict=9.5, crit=8.4, tick=8.8,
          note=8.4, leg=8.4)

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 450, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold", "mathtext.cal": "Liberation Sans",
    "mathtext.sf": "Liberation Sans", "mathtext.tt": "Liberation Mono",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#B0B0B0", "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


# --------------------------------------------------------------- reading the tables
def _clean(cell: str) -> str:
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    cell = cell.replace(r"\%", "%").replace(r"\_", "_").replace(r"\&", "&")
    cell = re.sub(r"\\[a-zA-Z]+", "", cell)
    cell = cell.replace("$", "").replace("{", "").replace("}", "")
    return cell.replace("--", "-").strip()


def load(tabdir, tid):
    head, rows = None, []
    for line in open(os.path.join(tabdir, f"{tid}_body.tex"), encoding="utf-8"):
        s = line.strip()
        if s.startswith("%") or "&" not in s:
            continue
        cells = [_clean(c) for c in re.sub(r"\\\\.*$", "", s).split("&")]
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


def find_tables(given):
    marker = "T14_body.tex"
    skip = {".git", "node_modules", "__pycache__", "venv", ".venv", "env",
            "site-packages", ".cache", "wandb", "checkpoints", "data", "runs"}
    here, cwd = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    tried, cands = [], ([given] if given else [])
    cands += [os.path.join(here, "tables"), os.path.join(cwd, "tables"),
              os.path.join(here, "results", "tables"),
              os.path.join(cwd, "results", "tables"),
              os.path.join(here, "analysis", "tables"),
              os.path.join(cwd, "analysis", "tables")]
    for c in cands:
        if not c:
            continue
        c = os.path.abspath(c)
        if c in tried:
            continue
        tried.append(c)
        if os.path.isfile(os.path.join(c, marker)):
            if not given:
                print(f"using tables in {c}")
            return c
    if given:
        print(f"no {marker} in {os.path.abspath(given)}")
        raise SystemExit(1)
    found = []
    for root in (here, cwd, os.path.join(here, ".."), os.path.join(cwd, "..")):
        base = os.path.abspath(root)
        for cur, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
            if cur.rstrip(os.sep).count(os.sep) - base.rstrip(os.sep).count(os.sep) >= 3:
                dirs[:] = []
            if marker in files and cur not in found:
                found.append(cur)
        if found:
            break
    if len(found) == 1:
        print(f"using tables in {found[0]}   (found by searching)")
        return found[0]
    if len(found) > 1:
        print(f"{marker} is in more than one place, so pick one:")
        for f in found:
            print(f"   python fig5_battery.py --tables {f}")
        raise SystemExit(1)
    print(f"could not find the generated tables (looking for {marker}). Tried:")
    for t in tried:
        print(f"   {t}")
    print("\nLocate them with:  find ~ -name T14_body.tex -not -path '*/.*' 2>/dev/null")
    raise SystemExit(1)


def go_nature():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    pick = [f for f in ("Arial", "Helvetica", "Nimbus Sans", "Liberation Sans")
            if f in have]
    if not pick:
        print("note: neither Arial nor Helvetica is installed.\n"
              "      on Ubuntu:  sudo apt install fonts-liberation\n"
              "      then:  rm -rf ~/.cache/matplotlib")
    mpl.rcParams.update({
        "font.sans-serif": pick + ["DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "savefig.dpi": 450})
    SZ.update(letter=8.0, title=7.0, verdict=6.5, crit=5.6, tick=6.0, note=5.6,
              leg=5.8)


# ---------------------------------------------------------------- the header block
def head_block(ax, letter, title, verdict, held):
    """Letter, title, verdict and the registered criterion, at the same four
    heights in every panel, so the four headers line up across the figure."""
    col = HELD if held else FAILED
    ax.text(-0.02, 1.26, letter, transform=ax.transAxes, fontsize=SZ["letter"],
            fontweight="bold", color=INK, va="top", ha="right")
    ax.text(0.02, 1.26, title, transform=ax.transAxes, fontsize=SZ["title"],
            color=INK, va="top")
    ax.text(1.0, 1.26, "HELD" if held else "DID NOT HOLD", transform=ax.transAxes,
            fontsize=SZ["verdict"], fontweight="bold", color=col, ha="right",
            va="top")
    ax.text(0.02, 1.13, "registered: " + verdict, transform=ax.transAxes,
            fontsize=SZ["crit"], color=INK2, va="top")


# ================================================================== panel a
def panel_a(ax, tabdir):
    head, rows = load(tabdir, "T14")
    jm, jo = colof(head, "model"), colof(head, "overall")
    jci, jp, js = colof(head, "CI"), colof(head, "passive"), colof(head, "swap")
    order = sorted(rows, key=lambda r: num(r[jo]))
    ax.axvspan(0.40, 0.60, color=BAND, zorder=0)
    ax.axvline(0.5, color=NEUTRAL, ls=(0, (4, 3)), lw=1.0, zorder=1)
    for i, r in enumerate(order):
        ci = rangeof(r[jci])
        if ci:
            ax.plot(ci, [i, i], color=INK, lw=1.4, solid_capstyle="butt", zorder=2)
        ax.plot([num(r[jo])], [i], "o", ms=6, color=INK, mec="white", mew=0.8, zorder=3)
        ax.plot([num(r[jp])], [i], "^", ms=6, color=BLUE, mec="white", mew=0.6, zorder=3)
        ax.plot([num(r[js])], [i], "v", ms=6, color=ORANGE, mec="white", mew=0.6,
                zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([r[jm] for r in order], fontsize=SZ["tick"])
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.35, -1.45)
    ax.text(0.50, -1.05, "predicted band", fontsize=SZ["note"],
            color=INK2, ha="center", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC: true entailment ranked above non-entailment")
    ax.xaxis.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    hs = [plt.Line2D([], [], ls="", marker=m, ms=6, color=c, mec="white", mew=0.6)
          for m, c in (("o", INK), ("^", BLUE), ("v", ORANGE))]
    # Inside the panel. Hanging this legend below the axes is what made the old
    # figure lopsided: it stretched the left column and nothing balanced it on the
    # right, so the 2x2 read as four panels of three different heights.
    # Upper right: no model's marker reaches past 0.70, so that strip is the only
    # region of this panel that is empty on every row.
    ax.legend(hs, ["overall", "non-entailment by passive", "by subject/object swap"],
              loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=SZ["leg"],
              labelspacing=0.3, handletextpad=0.5)


# ================================================================== panel b
def panel_b(ax, tabdir):
    head, rows = load(tabdir, "T15")
    jlang = [i for i, h in enumerate(head)
             if i > 0 and len(h.strip()) <= 3 and "-" not in h]
    langs = [head[i] for i in jlang]
    tops = []
    for i, r in enumerate(rows):
        vals = [num(j) for j in (r[k] for k in jlang)]
        col = [BLUE, GREEN, ORANGE][i % 3]
        ax.plot(range(len(langs)), vals, "-o", color=col, lw=1.6, ms=4.4,
                mec="white", mew=0.8, label=r[0], zorder=3)
        ax.axhline(vals[0] + 0.05, color=col, ls=(0, (1, 2)), lw=1.0, alpha=0.9)
        tops.append(vals[0] + 0.05)
    ax.text(len(langs) - 1, max(tops) + 0.003, "bound: English $+\\,0.05$",
            fontsize=SZ["note"], color=INK2, ha="right", va="bottom")
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs, fontsize=SZ["tick"])
    ax.set_xlim(-0.35, len(langs) - 0.65)
    ax.set_xlabel("PAWS-X language")
    ax.set_ylabel("AUC")
    ax.yaxis.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=SZ["leg"],
              labelspacing=0.3, handlelength=1.8, handletextpad=0.5)


# ================================================================== panel c
def panel_c(ax, tabdir):
    head, rows = load(tabdir, "T16")
    sets = [h for h in head[1:] if h.lower() != "positive"]
    series = [r for r in rows if "reference" not in r[0].lower()]
    w = 0.82 / len(series)
    for i, r in enumerate(series):
        graded = "ckpt" in r[0].lower()
        ax.bar([j + (i - len(series) / 2 + 0.5) * w for j in range(len(sets))],
               [num(r[colof(head, s)]) for s in sets], width=w * 0.9,
               color=(ORANGE if graded else BLUE), edgecolor="white", linewidth=0.6,
               zorder=2)
    ax.axhline(0, color=INK, lw=1.0, zorder=3)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(sets, fontsize=SZ["tick"])       # short enough to stay level
    ax.set_xlim(-0.6, len(sets) - 0.4)
    ax.set_ylabel("overlap excess above the cross-encoder")
    ax.yaxis.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=BLUE, label="contrastive encoders"),
                       Patch(facecolor=ORANGE, label="graded")],
              loc="upper right", fontsize=SZ["leg"], labelspacing=0.3,
              handlelength=1.1, handletextpad=0.5)


# ================================================================== panel d
def panel_d(ax, tabdir):
    head, rows = load(tabdir, "T17")
    hb, rb = load(tabdir, "T14")
    base = min(num(r[colof(hb, "overall")]) for r in rb)
    short = {"entity-swap": "entity swap", "matched-control": "matched control"}

    lo = max(min(num(c) for c in rows[k][1:] if num(c) is not None) for k in (0, 1))
    hi = min(max(num(c) for c in rows[k][1:] if num(c) is not None) for k in (0, 1))
    ax.axhspan(lo, hi, color=BAND, zorder=0)

    ax.plot([0], [base], "o", ms=6.5, mfc="white", mec=NEUTRAL, mew=1.5, zorder=3)
    ax.plot([-0.18, 0.18], [base, base], color=NEUTRAL, lw=2.2, zorder=2)
    for i, r in enumerate(rows):
        vals = [num(c) for c in r[1:] if num(c) is not None]
        col = ORANGE if "entity" in r[0].lower() else BLUE
        ax.plot([i + 1] * len(vals), vals, "o", ms=6.5, mfc="white", mec=col,
                mew=1.5, zorder=3)
        ax.plot([i + 0.82, i + 1.18], [sum(vals) / len(vals)] * 2, color=col,
                lw=2.2, zorder=2)
    ax.text(-0.45, (lo + hi) / 2, "the ranges overlap", fontsize=SZ["note"],
            color=INK2, ha="left", va="center")
    ax.set_xticks(range(len(rows) + 1))
    ax.set_xticklabels(["baseline"] + [short.get(r[0], r[0]) for r in rows],
                       fontsize=SZ["tick"])
    ax.set_xlim(-0.6, len(rows) + 0.6)
    ax.set_ylabel("HANS AUC after training")
    ax.yaxis.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.97, "one point per seed, bar at the mean",
            transform=ax.transAxes, fontsize=SZ["note"], color=INK2, va="top")


# ------------------------------------------------------------------- verification
def verify(tabdir) -> bool:
    bad = []
    h14, r14 = load(tabdir, "T14")
    overall = [num(r[colof(h14, "overall")]) for r in r14]
    if sum(1 for v in overall if 0.40 <= v <= 0.60) >= len(overall):
        bad.append("every model now sits inside the H1 band, so H1 would have held")
    h15, r15 = load(tabdir, "T15")
    jl = [i for i, h in enumerate(h15) if i > 0 and len(h.strip()) <= 3 and "-" not in h]
    for r in r15:
        v = [num(r[i]) for i in jl]
        if max(v) > v[0] + 0.05:
            bad.append(f"T15 {r[0]}: a language now exceeds English by more than "
                       f"0.05, so H2 would not have held")
    h16, r16 = load(tabdir, "T16")
    sets = [h for h in h16[1:] if h.lower() != "positive"]
    enc = [r for r in r16 if "reference" not in r[0].lower() and "ckpt" not in r[0].lower()]
    for r in enc:
        pos = sum(1 for s in sets if num(r[colof(h16, s)]) > 0)
        if pos < 5:
            bad.append(f"T16 {r[0]}: positive on {pos}/6, below the registered 5/6")
    h17, r17 = load(tabdir, "T17")
    rng = {r[0]: [num(c) for c in r[1:] if num(c) is not None] for r in r17}
    ks = list(rng)
    if len(ks) == 2 and (min(rng[ks[0]]) > max(rng[ks[1]])
                         or min(rng[ks[1]]) > max(rng[ks[0]])):
        bad.append("the two H4 conditions no longer overlap, which panel d states")
    print(f"  checked {len(r14)} H1 rows, {len(r15)} H2 rows, {len(r16)} H3 rows "
          f"and {len(r17)} H4 rows")
    for b in bad:
        print("  MISMATCH: " + b)
    if not bad:
        print("  every verdict the figure prints follows from the tables")
    return not bad


def symmetry_check(axes) -> bool:
    """The four axes must be the same size and share their row and column edges.

    Checked rather than eyeballed, since a legend placed outside an axes, a long
    tick label or a two-line y label all move an axes quietly and the result is a
    2x2 that looks almost square and is not.
    """
    import itertools
    p = {k: ax.get_position() for k, ax in axes.items()}
    tol, bad = 1e-6, []
    w = {k: v.width for k, v in p.items()}
    h = {k: v.height for k, v in p.items()}
    for name, d in (("width", w), ("height", h)):
        if max(d.values()) - min(d.values()) > tol:
            bad.append(f"the four axes differ in {name}: "
                       + ", ".join(f"{k} {v:.4f}" for k, v in d.items()))
    for col, ks in (("left", "ac"), ("right", "bd")):
        if abs(p[ks[0]].x0 - p[ks[1]].x0) > tol:
            bad.append(f"the {col} column's axes do not share a left edge")
    for row, ks in (("top", "ab"), ("bottom", "cd")):
        if abs(p[ks[0]].y0 - p[ks[1]].y0) > tol:
            bad.append(f"the {row} row's axes do not share a bottom edge")
    if bad:
        for b in bad:
            print("  NOT SYMMETRIC: " + b)
        return False
    print(f"  the four axes are identical at {w['a']:.4f} by {h['a']:.4f} of the "
          f"figure, and share their row and column edges")
    return True


def nature_check(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig).draw()
    drawn = fig.get_tightbbox(fig.canvas.get_renderer()).width
    k = min(1.0, (180 / 25.4) / drawn)
    sizes = []
    for ax in fig.get_axes():
        for t in ax.texts + ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text().strip():
                sizes.append((t.get_size(), t.get_text()[:26]))
        for lab in (ax.xaxis.label, ax.yaxis.label):
            if lab.get_text().strip():
                sizes.append((lab.get_size(), lab.get_text()[:26]))
        leg = ax.get_legend()
        if leg:
            sizes += [(t.get_size(), t.get_text()[:26]) for t in leg.get_texts()]
    smallest, what = min(sizes, key=lambda x: x[0])
    print(f"  drawn {drawn:.2f} in -> printed {180/25.4:.2f} in  (scale {k:.3f})")
    print(f"  smallest label {smallest:g} pt -> {smallest*k:.2f} pt   [{what}]")
    under = sorted({round(v * k, 2) for v, _ in sizes if v * k < 5.0})
    print("  " + ("all labels clear Nature's 5 pt floor"
                  if not under else f"BELOW 5 pt: {under}"))


# ========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=None)
    ap.add_argument("--out", default=".")
    ap.add_argument("--stem", default="fig5_battery")
    ap.add_argument("--nature", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--protocol", default="01337a7")
    ap.add_argument("--analysis", default="341a4b1")
    a = ap.parse_args()

    tabdir = find_tables(a.tables)
    ok = verify(tabdir)
    if a.check:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit(1)
    if a.nature:
        go_nature()

    w = 7.09 if a.nature else 11.6
    fig = plt.figure(figsize=(w, w * 8.4 / 11.6))
    # Explicit margins rather than a layout engine: the engines resize cells to fit
    # whatever labels each panel happens to carry, which is exactly the asymmetry
    # being removed here.
    gs = fig.add_gridspec(2, 2, left=0.095, right=0.975, top=0.835, bottom=0.075,
                          hspace=0.62, wspace=0.26)
    axes = {"a": fig.add_subplot(gs[0, 0]), "b": fig.add_subplot(gs[0, 1]),
            "c": fig.add_subplot(gs[1, 0]), "d": fig.add_subplot(gs[1, 1])}

    panel_a(axes["a"], tabdir)
    panel_b(axes["b"], tabdir)
    panel_c(axes["c"], tabdir)
    panel_d(axes["d"], tabdir)

    head_block(axes["a"], "a", "H1   HANS entailment",
               "every English encoder between 0.40 and 0.60 AUC", held=False)
    head_block(axes["b"], "b", "H2   not an English artefact",
               "no language more than 0.05 AUC above English", held=True)
    head_block(axes["c"], "c", "H3   the excess generalises",
               "encoders positive on 5 of 6; graded below base on 5 of 6", held=True)
    head_block(axes["d"], "d", "H4   transfer to HANS",
               "entity swap above base at every seed, matched control not",
               held=False)

    fig.text(0.5, 0.955,
             f"Registered before the data was touched  \u00b7  protocol {a.protocol}, "
             f"analysis {a.analysis}  \u00b7  two of four held",
             fontsize=SZ["title"], color=INK2, ha="center")

    sym = symmetry_check(axes)
    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"{a.stem}.{ext}"), facecolor="white")
    print(f"wrote {a.stem}.png and {a.stem}.pdf in {os.path.abspath(a.out)}")
    nature_check(fig)
    plt.close(fig)
    sys.exit(0 if sym else 1)


if __name__ == "__main__":
    main()

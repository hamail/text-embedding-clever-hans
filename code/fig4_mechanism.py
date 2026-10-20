#!/usr/bin/env python3
"""Figure 4 -- why no edit of the frozen representation repairs the failure.

Writes fig4_mechanism.png and .pdf, the names \\includegraphics already asks for.

  a  a role rides on the entity it belongs to          (T09)
  b  the rank of the binding term climbs with the vocabulary  (T13)
  c  removing the cue costs more for role swaps        (T10, ten matched cells)

Same three panels as before.  What changes is the typography and the palette, which
now match Figures 2 and 3: Okabe-Ito colours, Arial or its metric twin embedded as
Type 42, type set at the size it prints, and a bold claim beside each panel letter
rather than a title above it.  No number moves.

Every value is read from the generated table bodies, and --check re-derives each
one against the cell it came from, exiting non-zero on any disagreement.

Run
    python fig4_mechanism.py                   # finds the tables itself
    python fig4_mechanism.py --nature          # 180 mm wide, Nature type sizes
    python fig4_mechanism.py --check           # verify only, draw nothing
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------- colour
# Okabe-Ito, the accessible palette Nature's figure guide points to (Wong, Nat.
# Methods 8, 441, 2011).  The pairs actually used were run through a lightness,
# chroma, colour-blind separation and contrast check; all pass.
BLUE = "#0072B2"      # Okabe-Ito blue        - encoders, and the signal bar in a
ORANGE = "#D55E00"    # Okabe-Ito vermillion  - Qwen3 decoders, and role-vs-role in a
GREEN = "#009E73"     # Okabe-Ito bluish green - the third model in b
PURPLE = "#AE5C85"    # Okabe-Ito reddish purple, darkened - the retrained model
INK = "#1A1A1A"
INK2 = "#555555"
GRID = "#D8D8D8"
NEUTRAL = "#8A8A8A"
GREY_MID = "#767676"  # a reference series in a, not a category of its own
GREY_PALE = "#C4C4C4"

# Point sizes used inside the panels.  They are absolute, therefore they do not
# follow rcParams when the figure is redrawn at Nature's width; go_nature() rewrites
# them so the small text actually shrinks instead of overflowing the page.
SZ = dict(letter=13.0, head=9.8, tick=8.5, note=7.8, leg=8.0)

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 450, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # TrueType, not matplotlib's Type 3
    "mathtext.fontset": "custom",            # keep maths in the same face as the text
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.cal": "Liberation Sans", "mathtext.sf": "Liberation Sans",
    "mathtext.tt": "Liberation Mono",   # all six, or matplotlib hunts for a cursive
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
    """Exact header match, with substring matching only as a fallback.

    Substring matching has produced a wrong column five separate times in this
    project, therefore it never gets the first try.
    """
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


def find_tables(given: str | None) -> str:
    """Locate the generated table bodies without a path hard-coded to one machine."""
    marker = "T09_body.tex"
    skip = {".git", "node_modules", "__pycache__", "venv", ".venv", "env",
            "site-packages", ".cache", "wandb", "checkpoints", "data", "runs"}
    here, cwd = os.path.dirname(os.path.abspath(__file__)), os.getcwd()
    tried, cands = [], ([given] if given else [])
    cands += [os.path.join(here, "tables"), os.path.join(cwd, "tables"),
              os.path.join(here, "results", "tables"),
              os.path.join(cwd, "results", "tables"),
              os.path.join(here, "analysis", "tables"),
              os.path.join(cwd, "analysis", "tables"),
              os.path.join(here, "..", "analysis", "tables")]
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
        for cur, dirs, files in os.walk(os.path.abspath(root)):
            dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
            if cur.rstrip(os.sep).count(os.sep) - \
               os.path.abspath(root).rstrip(os.sep).count(os.sep) >= 3:
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
            print(f"   python fig4_mechanism.py --tables {f}")
        raise SystemExit(1)
    print(f"could not find the generated tables (looking for {marker}). Tried:")
    for t in tried:
        print(f"   {t}")
    print("\nLocate them with:  find ~ -name T09_body.tex -not -path '*/.*' 2>/dev/null")
    print("then pass that folder with --tables")
    raise SystemExit(1)


def go_nature() -> None:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    pick = [f for f in ("Arial", "Helvetica", "Nimbus Sans", "Liberation Sans")
            if f in have]
    if not pick:
        print("note: neither Arial nor Helvetica is installed, so the figure falls "
              "back to the default sans face.\n"
              "      on Ubuntu:  sudo apt install fonts-liberation\n"
              "      then clear the stale cache:  rm -rf ~/.cache/matplotlib")
    mpl.rcParams.update({
        "font.sans-serif": pick + ["DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "savefig.dpi": 450,
    })
    SZ.update(letter=8.0, head=7.0, tick=6.0, note=5.6, leg=5.8)


# ------------------------------------------------------------------ reading T10
def matched_cells(tabdir):
    """The ten cells where an entity edit and a control edit share rank and lambda.

    The ratio the Results quote is per cell.  Pooling the edits first and dividing
    once gives a different statistic, therefore the pairing happens before the
    division and a cell with no partner is dropped rather than averaged in.
    """
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
        ctrl = next((k for k in per if k[0] == m and "control" in k[1].lower()
                     and k[2] == rk and k[3] == lam), None)
        if ctrl:
            cells.append((m, rk, lam, v / per[ctrl]))
    cells.sort(key=lambda t: t[3])
    return cells


# ================================================================== panel a
def panel_a(ax, tabdir):
    """One signal against three references, rather than four peer categories.

    Two blues cannot both clear the contrast floor and stay far enough apart to be
    told apart, so the references do not each get a hue.  The alignment with the
    entity's own identity carries the claim and takes the strong colour; role
    against role takes the second, since the prose contrasts it with random; the
    remaining two references are greys.
    """
    head, rows = load(tabdir, "T09")
    jm = colof(head, "model")
    series = [("own name", BLUE, "with its own entity"),
              ("role vs role", ORANGE, "with other role directions"),
              ("other names", GREY_MID, "with other entities"),
              ("random level", GREY_PALE, "random directions")]
    n, w = len(rows), 0.19
    for s, (key, col, lab) in enumerate(series):
        j = colof(head, key)
        ax.bar([i + (s - 1.5) * w for i in range(n)], [num(r[j]) for r in rows],
               width=w * 0.92, color=col, label=lab,
               edgecolor="white", linewidth=0.8, zorder=2)
    ax.set_xticks(range(n))
    ax.set_xticklabels([r[jm] for r in rows], rotation=14, ha="right",
                       fontsize=SZ["tick"])
    ax.set_ylabel("alignment with a reference direction, $|\\cos|$")
    ax.set_ylim(0, 0.78)
    ax.yaxis.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=SZ["leg"], labelspacing=0.3,
              handlelength=1.3, handletextpad=0.5)


# ================================================================== panel b
def panel_b(ax, tabdir):
    head, rows = load(tabdir, "T13")
    sizes = [int(num(h)) for h in head[1:]]
    models = [r for r in rows if "generic" not in r[0].lower()]
    for i, r in enumerate(models):
        ax.plot(sizes, [num(c) for c in r[1:]], "-o",
                color=[BLUE, GREEN, ORANGE][i % 3], lw=1.6, ms=4.4,
                mec="white", mew=0.9, label=r[0], zorder=3)
    ax.axhline(1, color=INK, ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text(sizes[0], 3.0, "a single shared role axis would need 1, at any size",
            fontsize=SZ["note"], color=INK, va="bottom")
    ax.set_xticks(sizes)
    ax.set_xlim(sizes[0] - 6, sizes[-1] + 6)
    ax.set_ylim(0, max(num(c) for r in models for c in r[1:]) * 1.22)
    ax.set_xlabel("entities in the vocabulary")
    ax.set_ylabel("directions holding $90\\%$\nof the binding term")
    ax.yaxis.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=SZ["leg"], labelspacing=0.3,
              handlelength=1.6, handletextpad=0.5)


# ================================================================== panel c
def panel_c(ax, cells):
    """Marker shape carries the model as well as colour.

    Blue against the darkened reddish purple sits in the 6 to 8 band for one form
    of colour blindness, which is allowed only alongside a second encoding, so the
    two models differ in shape too.
    """
    mods, shapes = [], ("o", "s", "D")
    for c in cells:
        if c[0] not in mods:
            mods.append(c[0])
    cols = [BLUE, PURPLE, ORANGE]
    lo, hi = cells[0][3], cells[-1][3]

    ax.axvspan(0, 1.0, color="#F2F2F2", zorder=0)
    for i, (m, rk, lam, ratio) in enumerate(cells):
        k = mods.index(m) % 3
        ax.plot([ratio], [i], marker=shapes[k], ms=6.0, color=cols[k],
                mec="white", mew=0.9, zorder=3)
        ax.text(ratio + 0.07, i, f"rank {rk}, $\\lambda$ {lam}",
                fontsize=SZ["note"], color=INK2, va="center")
    ax.axvline(1.0, color=INK, ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text(0.94, len(cells) - 0.4, "equal cost", fontsize=SZ["note"], color=INK,
            ha="right", va="center")

    # Sorting by ratio happens to stack each model's cells together, so each model
    # is named once beside its own block instead of in a legend box, which has no
    # free corner here: every corner either holds a point or holds its label.
    span = {m: [i for i, c in enumerate(cells) if c[0] == m] for m in mods}
    contiguous = all(ix == list(range(ix[0], ix[-1] + 1)) for ix in span.values())
    if contiguous:
        for i, m in enumerate(mods):
            ix = span[m]
            ax.text(0.10, sum(ix) / len(ix), m, fontsize=SZ["leg"],
                    color=cols[i % 3], fontweight="bold", va="center", ha="left")
    else:
        hs = [plt.Line2D([], [], ls="", marker=shapes[i % 3], ms=6.0,
                         color=cols[i % 3], mec="white", mew=0.9)
              for i in range(len(mods))]
        ax.legend(hs, mods, loc="lower left", bbox_to_anchor=(0.01, 0.02),
                  fontsize=SZ["leg"], labelspacing=0.3, handletextpad=0.5)
    ax.set_yticks([])
    ax.set_ylim(-0.9, len(cells) + 0.1)
    ax.set_xlim(0, max(3.7, hi + 0.8))
    ax.set_xlabel("STS-B lost per unit of swap similarity removed,\n"
                  "role swaps \u00f7 swaps carrying no role")
    ax.xaxis.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def head_label(ax, letter, claim, color, dx, dy, dxc):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=SZ["letter"],
            fontweight="bold", color=INK, va="bottom")
    ax.text(dxc, dy, claim, transform=ax.transAxes, fontsize=SZ["head"],
            fontweight="bold", color=color, va="bottom")


# ------------------------------------------------------------------- verification
def verify(tabdir, cells) -> bool:
    bad = []
    head, rows = load(tabdir, "T09")
    own = [num(r[colof(head, "own name")]) for r in rows]
    rvr = [num(r[colof(head, "role vs role")]) for r in rows]
    rnd = [num(r[colof(head, "random level")]) for r in rows]
    oth = [num(r[colof(head, "other names")]) for r in rows]
    if len(rows) != 3:
        bad.append(f"T09 should hold three models, it holds {len(rows)}")
    if not min(own) > max(rvr + rnd + oth):
        bad.append("in T09 the alignment with a role's own entity is no longer above "
                   "every reference, which is panel a's whole claim")

    head13, rows13 = load(tabdir, "T13")
    generic = [r for r in rows13 if "generic" in r[0].lower()]
    models = [r for r in rows13 if "generic" not in r[0].lower()]
    if not generic or set(generic[0][1:]) != {"1"}:
        bad.append("T13's generic-axis row no longer reads 1 at every vocabulary size")
    for r in models:
        v = [num(c) for c in r[1:]]
        if not all(v[i] < v[i + 1] for i in range(len(v) - 1)):
            bad.append(f"T13 {r[0]}: the rank no longer climbs at every step, so "
                       f"panel b's claim does not hold for this model")

    if len(cells) != 10:
        bad.append(f"T10 yields {len(cells)} matched cells, not ten")
    below = [c for c in cells if c[3] <= 1.0]
    if below:
        bad.append(f"{len(below)} matched cells are at or below equal cost, which "
                   f"panel c claims none are")

    print(f"  checked {len(rows)} alignment rows, {len(models)} rank series and "
          f"{len(cells)} matched cells")
    if bad:
        print("  MISMATCH between the figure and the tables:")
        for b in bad:
            print("    " + b)
        return False
    print("  every plotted value traces to a table cell")
    return True


def report_numbers(tabdir, cells) -> None:
    head, rows = load(tabdir, "T09")
    g = lambda k: [num(r[colof(head, k)]) for r in rows]
    head13, rows13 = load(tabdir, "T13")
    models = [r for r in rows13 if "generic" not in r[0].lower()]
    last = [num(r[-1]) for r in models]
    unsup = next((r for r in models if "unsup" in r[0].lower()), models[0])
    print("  for the caption and the prose:")
    print(f"    a  with its own entity        {min(g('own name')):.3f}"
          f"-{max(g('own name')):.3f}")
    print(f"    a  with other role directions {min(g('role vs role')):.3f}"
          f"-{max(g('role vs role')):.3f}")
    print(f"    a  with other entities        {min(g('other names')):.3f}"
          f"-{max(g('other names')):.3f}")
    print(f"    a  random level               {min(g('random level')):.3f}"
          f"-{max(g('random level')):.3f}")
    print(f"    b  rank at the largest vocabulary  {min(last):.0f}-{max(last):.0f}")
    print(f"    b  {unsup[0]} across the range     "
          f"{', '.join(c for c in unsup[1:])}")
    inc = [num(unsup[i + 1]) - num(unsup[i]) for i in range(1, len(unsup) - 1)]
    print(f"    b  its increments                  "
          f"{', '.join(f'+{d:.0f}' for d in inc)}")
    print(f"    c  matched cells                   {len(cells)}")
    print(f"    c  ratio                           {cells[0][3]:.4f} to "
          f"{cells[-1][3]:.4f}  -> the prose must say "
          f"{cells[0][3]:.1f} to {cells[-1][3]:.1f}")


def nature_check(fig) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig).draw()
    drawn_in = fig.get_tightbbox(fig.canvas.get_renderer()).width
    printed_in = 180 / 25.4
    k = min(1.0, printed_in / drawn_in)
    sizes = []
    for ax in fig.get_axes():
        for t in ax.texts + ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text().strip():
                sizes.append((t.get_size(), t.get_text()[:28]))
        for lab in (ax.xaxis.label, ax.yaxis.label):
            if lab.get_text().strip():
                sizes.append((lab.get_size(), lab.get_text()[:28]))
        leg = ax.get_legend()
        if leg:
            sizes += [(t.get_size(), t.get_text()[:28]) for t in leg.get_texts()]
    if not sizes:
        return
    smallest, what = min(sizes, key=lambda x: x[0])
    print(f"  drawn {drawn_in:.2f} in -> printed {printed_in:.2f} in  (scale {k:.3f})")
    print(f"  smallest label {smallest:g} pt -> {smallest * k:.2f} pt   [{what}]")
    under = sorted({round(v * k, 2) for v, _ in sizes if v * k < 5.0})
    print("  " + ("all labels clear Nature's 5 pt floor"
                  if not under else f"BELOW 5 pt: {under}"))


# ========================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=None)
    ap.add_argument("--out", default=".")
    ap.add_argument("--stem", default="fig4_mechanism",
                    help="output basename; the manuscript asks for fig4_mechanism")
    ap.add_argument("--nature", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    a.tables = find_tables(a.tables)
    cells = matched_cells(a.tables)
    if not cells:
        print("no matched cells in T10: no entity edit shares a rank and lambda "
              "with a control edit, therefore panel c cannot be drawn")
        sys.exit(1)

    ok = verify(a.tables, cells)
    report_numbers(a.tables, cells)
    if a.check:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit(1)

    if a.nature:
        go_nature()

    w = 7.09 if a.nature else 12.6
    fig = plt.figure(figsize=(w, w * 4.4 / 12.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.12], wspace=0.34)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    panel_a(ax_a, a.tables)
    panel_b(ax_b, a.tables)
    panel_c(ax_c, cells)

    head_label(ax_a, "a", "A role rides on its entity", BLUE,
               dx=-0.21, dy=1.07, dxc=-0.155)
    head_label(ax_b, "b", "So the rank climbs", BLUE,
               dx=-0.21, dy=1.07, dxc=-0.155)
    head_label(ax_c, "c", "and costs more to undo", ORANGE,
               dx=-0.07, dy=1.07, dxc=-0.025)

    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"{a.stem}.{ext}"), facecolor="white")
    print(f"wrote {a.stem}.png and {a.stem}.pdf in {os.path.abspath(a.out)}")
    nature_check(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()

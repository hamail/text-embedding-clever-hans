#!/usr/bin/env python3
"""Figure 3 -- the dissociation between what is recorded and what is used.

Writes fig3_dissociation.png and .pdf, the names \\includegraphics already asks for.

  a  every model, who acts on whom read out in both voices   (T03)
  b  a readout trained on one voice is a position code       (T03, calibration rows)
  c  the record survives at every depth                      (T04)
  d  no layer's similarity separates a reversal              (T04)

Panels a and b carry the eleven-model probe, which used to sit only in the supplementary
material.  Panels c and d are the layer sweep, overlaid on a shared relative-depth axis
rather than split into four cells, so the figure says "two architectures swept in depth"
instead of implying that only two models were tested at all.

Every number is read from the generated table bodies.  Nothing is typed in by hand, and
--check re-reads each plotted value against the cell it came from and exits non-zero on
any disagreement, so the figure cannot drift away from the tables unnoticed.

Run
    python fig3_dissociation.py                   # finds the tables, draws at screen size
    python fig3_dissociation.py --nature          # 180 mm wide, Nature type sizes
    python fig3_dissociation.py --check           # verify only, draw nothing
    python fig3_dissociation.py --tables <dir> --out <dir>
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
# Methods 8, 441, 2011).  Checked for lightness band, chroma floor, colour-blind
# separation of every adjacent pair, and 3:1 contrast on white; all pass.
BLUE = "#0072B2"      # Okabe-Ito blue        - the seven off-the-shelf encoders
ORANGE = "#D55E00"    # Okabe-Ito vermillion  - the Qwen3 decoders
PURPLE = "#AE5C85"    # Okabe-Ito reddish purple, darkened to clear 3:1 on white
GREEN = "#009E73"     # reserved for "not blind" in Figure 2; unused here
INK = "#1A1A1A"
INK2 = "#555555"
GRID = "#D8D8D8"
NEUTRAL = "#8A8A8A"
FLOOR = "#DCDCDC"     # the control-task floor in a
CORRIDOR = "#C9C9C9"  # the three calibration readouts in b
ABOVE = "#F0F0F0"     # the region a meaning-sensitive similarity would occupy, in d

# Point sizes used inside the panels.  They are absolute, so they do not follow
# rcParams when the figure is redrawn at Nature's width; go_nature() rewrites them.
SZ = dict(letter=13.0, head=9.8, model=8.4, note=7.8, leg=8.0)

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 450, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # TrueType, not matplotlib's Type 3
    "mathtext.fontset": "custom",            # keep maths in the same face as the text
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
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

QWEN = ("Qwen3-0.6B", "Qwen3-4B", "Qwen3-8B")
RETRAINED = ("retrained",)
FAM_COLOR = {"bi": BLUE, "qwen": ORANGE, "retrained": PURPLE}
FAM_LABEL = {"bi": "sentence encoders (7)",
             "qwen": "Qwen3 decoders (3)",
             "retrained": "retrained with reversals (1)"}


def family(name: str) -> str:
    if name in QWEN:
        return "qwen"
    if name in RETRAINED:
        return "retrained"
    return "bi"


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
    project, so it never gets the first try.
    """
    for i, h in enumerate(head):
        if h.strip().lower() == key.strip().lower():
            return i
    for i, h in enumerate(head):
        if key.lower() in h.lower():
            return i
    raise SystemExit(f"no column matching {key!r} in {head}")


def col_exact(head, key):
    """Match on the header with every space removed, and no fallback at all.

    _clean strips the \\rightarrow macros, so "act$\\rightarrow$act" arrives as
    "actact".  Those four column names are near-anagrams of one another, so a
    substring fallback here would be a liability rather than a convenience.
    """
    want = key.replace(" ", "").lower()
    for i, h in enumerate(head):
        if h.replace(" ", "").lower() == want:
            return i
    raise SystemExit(f"no column {key!r} in {head}")


def num(cell):
    s = cell.replace("%", "").strip()
    if s in ("-", "--", "", "nan"):
        return None
    s = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " ", s)
    m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m[0]) if m else None


MARKER = "T03_body.tex"          # the file that identifies a tables directory
SKIP = {".git", "node_modules", "__pycache__", ".ipynb_checkpoints", "venv",
        ".venv", "env", "site-packages", ".cache", ".mypy_cache", "wandb",
        "checkpoints", "runs", "models", "data", "datasets", "outputs_raw"}


def _scan(root: str, depth: int, found: list, seen: set) -> None:
    """Walk at most `depth` levels below root looking for the marker file.

    Bounded rather than exhaustive, and heavy directories are skipped, so this
    stays fast on a working repository instead of crawling a checkpoint folder.
    """
    root = os.path.abspath(root)
    if root in seen or not os.path.isdir(root):
        return
    seen.add(root)
    base = root.rstrip(os.sep).count(os.sep)
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
        if cur.rstrip(os.sep).count(os.sep) - base >= depth:
            dirs[:] = []
        if MARKER in files and cur not in found:
            found.append(cur)


def find_tables(given: str | None) -> str:
    """Locate the generated table bodies without a path hard-coded to one machine.

    The named places are tried first.  Failing those, the script directory, the
    working directory and their parents are scanned a few levels down, because the
    tables are generated into whatever folder the analysis happens to live in and
    that folder is not the same on two machines.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    tried, cands = [], ([given] if given else [])
    cands += [
        os.path.join(here, "tables"),
        os.path.join(cwd, "tables"),
        os.path.join(here, "analysis", "tables"),
        os.path.join(cwd, "analysis", "tables"),
        os.path.join(here, "..", "analysis", "tables"),
        os.path.join(here, "..", "tables"),
        os.path.join(cwd, "..", "tables"),
        os.path.join(here, "..", "..", "analysis", "tables"),
    ]
    for c in cands:
        if not c:
            continue
        c = os.path.abspath(c)
        if c in tried:                       # the script dir and cwd are often the same
            continue
        tried.append(c)
        if os.path.isfile(os.path.join(c, MARKER)):
            if not given:
                print(f"using tables in {c}")
            return c

    if given:                                # an explicit path that was wrong: say so
        print(f"no {MARKER} in {os.path.abspath(given)}")
        raise SystemExit(1)

    found, seen = [], set()
    for root, depth in ((here, 3), (cwd, 3),
                        (os.path.join(here, ".."), 3),
                        (os.path.join(cwd, ".."), 3),
                        (os.path.join(here, "..", ".."), 2)):
        _scan(root, depth, found, seen)
        if found:
            break
    if len(found) == 1:
        n = len([f for f in os.listdir(found[0]) if re.fullmatch(r"T\d+\w*_body\.tex", f)])
        print(f"using tables in {found[0]}")
        print(f"   found by searching rather than at a known path, and it holds {n} "
              f"table bodies -- check that is the folder you meant")
        return found[0]
    if len(found) > 1:
        print(f"found {MARKER} in more than one place, so pick one:")
        for f in found:
            print(f"   python fig3_dissociation.py --tables {f}")
        raise SystemExit(1)

    print("could not find the generated tables (looking for T03_body.tex).")
    print("Named places tried:")
    for t in tried:
        print(f"   {t}")
    print(f"and nothing turned up within three levels of {here} or {cwd}.")
    print("\nLocate them with:")
    print("   find ~ -name T03_body.tex -not -path '*/.*' 2>/dev/null")
    print("then pass that folder:")
    print("   python fig3_dissociation.py --tables <the folder it printed>")
    print("It is the folder holding T01_body.tex ... T19_body.tex, the same one "
          "make_figures.py reads.")
    raise SystemExit(1)


def go_nature() -> None:
    """Draw at 180 mm with type at the size it will print, rather than scaling down."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    pick = [f for f in ("Arial", "Helvetica", "Nimbus Sans", "Liberation Sans")
            if f in have]
    if not pick:
        print("note: neither Arial nor Helvetica is installed, so the figure falls back "
              "to the default sans face.\n"
              "      on Ubuntu:  sudo apt install fonts-liberation   "
              "(metric-compatible with Arial)\n"
              "      then clear the stale cache:  rm -rf ~/.cache/matplotlib")
    mpl.rcParams.update({
        "font.sans-serif": pick + ["DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "savefig.dpi": 450,
    })
    SZ.update(letter=8.0, head=7.0, model=6.0, note=5.6, leg=5.8)


# --------------------------------------------------------------- T03, split in two
def read_probe(tabdir):
    """The three calibration readouts and the eleven model readouts, separately."""
    head, rows = load(tabdir, "T03")
    j = {k: col_exact(head, v) for k, v in
         (("name", "features"), ("aa", "actact"), ("ap", "actpas"),
          ("ba", "bothact"), ("bp", "bothpas"),
          ("cs", "control seen"), ("cn", "control new"))}
    cal, mods = {}, []
    for r in rows:
        rec = {k: (r[i] if k == "name" else num(r[i])) for k, i in j.items()}
        if rec["name"].startswith("calibration:"):
            cal[rec["name"].split(":", 1)[1].strip()] = rec
        else:
            mods.append(rec)
    return cal, mods


def read_sweep(tabdir):
    head, rows = load(tabdir, "T04")
    jm, jl = colof(head, "model"), colof(head, "layer")
    ja, jp = colof(head, "probe active"), colof(head, "probe passive")
    jr = colof(head, "distance ratio")
    out = {}
    for r in rows:
        out.setdefault(r[jm], []).append(
            (int(num(r[jl])), num(r[ja]), num(r[jp]), num(r[jr])))
    for k in out:
        out[k].sort()
    return out


def rel(layers):
    """Depth as a fraction of the network, so a twelve-layer and a twenty-eight-layer
    model can share one axis.  Without it the two sweeps cannot be overlaid, and the
    figure has to spend four cells saying what two can say."""
    n = max(layers)
    return [l / n for l in layers]


# ================================================================== panel a
def panel_a(ax, cal, mods):
    """Eleven models, the role read out in both voices, against the control floor.

    The axis stops at 90 per cent.  Where a role code would land is panel b's
    business; this panel asks only whether the readout clears the floor, therefore
    the floor and the readouts should fill the width rather than share it with empty
    space.
    """
    # Group by family rather than inheriting the table's row order, keeping the
    # table's own order inside each family.  Two builds of T03 have shipped with the
    # models in different sequences, and when the order is inherited the purple and
    # orange rows land inside the blue block, so the colour coding fights the layout
    # and the figure changes shape depending on which build it happens to read.
    rank = {"bi": 0, "retrained": 1, "qwen": 2}
    seq = sorted(enumerate(mods), key=lambda t: (rank[family(t[1]["name"])], t[0]))
    order = [m for _, m in reversed(seq)]             # first drawn row at the top
    ys = list(range(len(order)))

    lo, hi = min(m["cn"] for m in mods), max(m["cn"] for m in mods)
    ax.axvspan(lo, hi, color=FLOOR, zorder=0)
    ax.axvline(50, color=NEUTRAL, ls=(0, (4, 3)), lw=0.9, zorder=1)

    for y, m in zip(ys, order):
        c = FAM_COLOR[family(m["name"])]
        ax.plot([m["bp"], m["ba"]], [y, y], color=c, lw=1.5, alpha=0.5,
                solid_capstyle="round", zorder=2)
        ax.plot(m["ba"], y, "o", color=c, ms=6.0, mec="white", mew=1.0, zorder=3)
        ax.plot(m["bp"], y, "s", color="white", ms=6.0, mec=c, mew=1.7, zorder=3)
        a, b = round(min(m["ba"], m["bp"])), round(max(m["ba"], m["bp"]))
        ax.text(max(m["ba"], m["bp"]) + 1.3, y,
                f"{a:.0f}" if a == b else f"{a:.0f}\u2013{b:.0f}",
                fontsize=SZ["note"], color=INK2, va="center", ha="left")

    ax.set_yticks(ys)
    ax.set_yticklabels([m["name"] for m in order], fontsize=SZ["model"])
    for t, m in zip(ax.get_yticklabels(), order):
        t.set_color(FAM_COLOR[family(m["name"])])
    ax.set_ylim(-0.75, len(order) - 0.25)
    ax.set_xlim(45, 93)
    ax.set_xticks([50, 60, 70, 80, 90])
    ax.set_xticklabels(["50%", "60%", "70%", "80%", "90%"])
    ax.set_xlabel("who acts on whom, read off by a linear probe")
    ax.xaxis.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    ax.annotate("control task, labels it never saw",
                xy=(hi + 0.3, -0.52), xytext=(hi + 3.2, -0.52),
                fontsize=SZ["note"], color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=NEUTRAL, lw=0.8,
                                shrinkA=2, shrinkB=0))

    ax.plot([], [], "o", color=NEUTRAL, ms=6.0, mec="white", mew=1.0,
            label="tested on active voice")
    ax.plot([], [], "s", color="white", ms=6.0, mec=NEUTRAL, mew=1.7,
            label="tested on passive voice")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              handletextpad=0.5, columnspacing=1.6, fontsize=SZ["leg"])


# ================================================================== panel b
def panel_b(ax, cal, mods):
    """act->act against act->pas, read against the three calibration readouts."""
    xa, xp = 0.0, 1.0

    # The calibration readouts go down first and wide, therefore they read as the
    # corridors the model lines are measured against rather than as three more models.
    for label in ("names by role", "names by order", "bag of names"):
        r = cal[label]
        ax.plot([xa, xp], [r["aa"], r["ap"]], color=CORRIDOR, lw=5.0,
                solid_capstyle="round", zorder=1)
        ax.plot([xa, xp], [r["aa"], r["ap"]], color=INK2, lw=1.0,
                ls=(0, (4, 2.5)), zorder=2)

    for m in mods:
        c = FAM_COLOR[family(m["name"])]
        ax.plot([xa, xp], [m["aa"], m["ap"]], color=c, lw=1.5, alpha=0.85,
                solid_capstyle="round", zorder=3)
        ax.plot([xa, xp], [m["aa"], m["ap"]], "o", color=c, ms=4.6,
                mec="white", mew=0.9, zorder=4)

    ax.text(xp + 0.08, 100, "names by role\nroles are what is stored",
            fontsize=SZ["note"], color=INK, va="center", linespacing=1.3)
    ax.text(xp + 0.08, 50, "bag of names\nnothing is stored",
            fontsize=SZ["note"], color=INK2, va="center", linespacing=1.3)
    ax.text(xp + 0.08, 2, "names by order\nposition is what is stored",
            fontsize=SZ["note"], color=INK2, va="center", linespacing=1.3)

    ax.set_xlim(-0.12, 1.98)
    ax.set_ylim(-6, 108)
    ax.set_xticks([xa, xp])
    ax.set_xticklabels(["trained and tested\non active voice",
                        "trained on active,\ntested on passive"], fontsize=SZ["leg"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("who acts on whom, read off\nby a linear probe")
    ax.yaxis.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(True)

    hs = [plt.Line2D([], [], color=FAM_COLOR[k], lw=1.6, marker="o", ms=4.6,
                     mec="white", mew=0.9) for k in ("bi", "qwen", "retrained")]
    ax.legend(hs, [FAM_LABEL[k] for k in ("bi", "qwen", "retrained")],
              loc="lower left", bbox_to_anchor=(-0.02, -0.02), labelspacing=0.3,
              handletextpad=0.5, fontsize=SZ["leg"])


def _depth_axis(ax):
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["input", "", "half", "", "output"])
    ax.set_xlabel("depth through the network")
    ax.yaxis.grid(True, alpha=0.55)
    ax.set_axisbelow(True)


# ================================================================== panel c
def panel_c(ax, sweep):
    hs = []
    for name, rowsn in sweep.items():
        c = FAM_COLOR[family(name)]
        L = rel([r[0] for r in rowsn])
        ax.plot(L, [r[1] for r in rowsn], "-", color=c, lw=1.5, marker="o", ms=3.4)
        ax.plot(L, [r[2] for r in rowsn], "--", color=c, lw=1.3, marker="s", ms=3.0,
                mfc="white", mew=0.9)
        hs.append((plt.Line2D([], [], color=c, lw=2.0), f"{name}  ({len(L)} layers)"))

    ax.axhline(50, color=NEUTRAL, ls=(0, (4, 3)), lw=0.9)
    ax.text(0.0, 50.6, "chance", fontsize=SZ["note"], color=NEUTRAL,
            va="bottom", ha="left")
    ax.set_ylim(45, 102)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_yticklabels(["50%", "60%", "70%", "80%", "90%", "100%"])
    ax.set_ylabel("who acts on whom, read off\nby a linear probe")
    _depth_axis(ax)

    voice = ax.legend(
        [plt.Line2D([], [], color=NEUTRAL, lw=1.5, marker="o", ms=3.4),
         plt.Line2D([], [], color=NEUTRAL, lw=1.3, ls="--", marker="s", ms=3.0,
                    mfc="white", mew=0.9)],
        ["tested on active voice", "tested on passive voice"],
        loc="upper left", bbox_to_anchor=(-0.01, 1.03), labelspacing=0.28,
        handletextpad=0.6, handlelength=2.4, fontsize=SZ["note"])
    ax.add_artist(voice)
    ax.legend([h for h, _ in hs], [t for _, t in hs],
              loc="lower right", bbox_to_anchor=(1.01, 0.115), labelspacing=0.28,
              handletextpad=0.6, handlelength=1.6, fontsize=SZ["note"])


# ================================================================== panel d
def panel_d(ax, sweep):
    ax.axhspan(1.0, 1.20, color=ABOVE, zorder=0)
    for name, rowsn in sweep.items():
        c = FAM_COLOR[family(name)]
        L = rel([r[0] for r in rowsn])
        ax.plot(L, [r[3] for r in rowsn], "-", color=c, lw=1.6, marker="o", ms=3.6,
                label=name, zorder=3)
    ax.axhline(1.0, color=INK, ls=(0, (4, 3)), lw=1.2, zorder=2)
    ax.text(0.5, 1.10, "where a similarity that tracked meaning would sit",
            fontsize=SZ["note"], color=INK, va="center", ha="center")
    ax.text(0.03, 0.80,
            "the record is read off at every one of\n"
            "these depths, and used at none of them",
            fontsize=SZ["note"], color=INK2, va="center", linespacing=1.35)
    ax.set_ylim(0, 1.20)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("distance to the reversal \u00f7\ndistance to the rewrite")
    _depth_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.60), labelspacing=0.28,
              handletextpad=0.6, handlelength=1.6, fontsize=SZ["note"])


def head_label(ax, letter, claim, color, dx, dy, dxc):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=SZ["letter"],
            fontweight="bold", color=INK, va="bottom")
    ax.text(dxc, dy, claim, transform=ax.transAxes, fontsize=SZ["head"],
            fontweight="bold", color=color, va="bottom")


# ------------------------------------------------------------------- verification
def verify(cal, mods, sweep) -> bool:
    """Re-read every drawn value against the cell it came from.

    A figure script that keeps drawing quietly after a table changes shape is how a
    stale number reaches a reviewer, therefore each assumption the four panels rest
    on is named here and the caller exits non-zero if any of them fails.
    """
    bad = []

    if len(mods) != 11:
        bad.append(f"T03 should hold eleven models, it holds {len(mods)}")
    for want in ("bag of names", "names by order", "names by role"):
        if want not in cal:
            bad.append(f"T03 calibration row {want!r} is missing")
    if cal.get("names by order", {}).get("aa") != 100.0 or \
       cal.get("names by order", {}).get("ap") != 0.0:
        bad.append("the order-code calibration no longer runs 100% -> 0% across "
                   "voices; panel b's lower corridor is drawn from it")
    if cal.get("names by role", {}).get("ap") != 100.0:
        bad.append("the role-code calibration no longer reads 100% across voices; "
                   "panel b's upper corridor is drawn from it")
    if cal.get("bag of names", {}).get("aa") != 50.0:
        bad.append("the bag-of-names calibration no longer reads 50%")

    for m in mods:
        n = m["name"]
        if not 0 <= m["ap"] <= 45:
            bad.append(f"{n}: act->pas {m['ap']}% is outside panel b's drawn range")
        if not 0 <= m["aa"] <= 108:
            bad.append(f"{n}: act->act {m['aa']}% is outside panel b's drawn range")
        for k, lab in (("ba", "both->act"), ("bp", "both->pas")):
            if not 45 <= m[k] <= 93:
                bad.append(f"{n}: {lab} {m[k]}% falls outside panel a's x limits")
        if not 45 <= m["cn"] <= 60:
            bad.append(f"{n}: control-new {m['cn']}% is outside the floor band")
        if m["cn"] >= min(m["ba"], m["bp"]):
            bad.append(f"{n}: the control floor is not below the readout, therefore "
                       f"panel a's claim does not hold for this row")

    nlay = {k: len(v) for k, v in sweep.items()}
    if nlay != {"unsup-SimCSE": 12, "Qwen3-0.6B": 28}:
        bad.append(f"T04's layer counts changed: {nlay}")
    for k, v in sweep.items():
        for layer, a, p, r in v:
            if not 0 <= r <= 1.0:
                bad.append(f"{k} layer {layer}: ratio {r} is off panel d's y axis")
            if not 45 <= a <= 102 or not 45 <= p <= 102:
                bad.append(f"{k} layer {layer}: probe {a}/{p}% is off panel c's y axis")

    n = len(mods) + len(cal) + sum(nlay.values())
    print(f"  checked {n} table rows behind the four panels")
    if bad:
        print("  MISMATCH between the figure and the tables:")
        for b in bad:
            print("    " + b)
        return False
    print("  every plotted value traces to a table cell")
    return True


def report_numbers(cal, mods, sweep) -> None:
    """Print the ranges the caption and the Results prose quote, so the two cannot
    drift apart without it showing up in the build log."""
    both = [v for m in mods for v in (m["ba"], m["bp"])]
    qw = [m for m in mods if family(m["name"]) == "qwen"]
    rt = [m for m in mods if family(m["name"]) == "retrained"][0]
    pr = [v for rows in sweep.values() for _, a, p, _ in rows for v in (a, p)]
    rr = [r for rows in sweep.values() for *_, r in rows]
    print("  for the caption and the prose:")
    print(f"    a  both-voice readout       {min(both):.1f}-{max(both):.1f}%")
    print(f"    a  control task, new labels {min(m['cn'] for m in mods):.1f}"
          f"-{max(m['cn'] for m in mods):.1f}%")
    print(f"    b  act->act                 {min(m['aa'] for m in mods):.1f}"
          f"-{max(m['aa'] for m in mods):.1f}%")
    print(f"    b  act->pas                 {min(m['ap'] for m in mods):.1f}"
          f"-{max(m['ap'] for m in mods):.1f}%")
    print(f"    b  act->pas, Qwen3 only     {min(m['ap'] for m in qw):.1f}"
          f"-{max(m['ap'] for m in qw):.1f}%")
    print(f"    b  act->pas, retrained      {rt['ap']:.1f}%   "
          f"(the highest of the eleven)")
    print(f"    c  probe over all layers    {min(pr):.1f}-{max(pr):.1f}%")
    print(f"    d  distance ratio           {min(rr):.3f}-{max(rr):.3f}")
    rank = {"bi": 0, "retrained": 1, "qwen": 2}
    drawn = [m["name"] for _, m in
             sorted(enumerate(mods), key=lambda t: (rank[family(t[1]["name"])], t[0]))]
    if drawn != [m["name"] for m in mods]:
        print("  note: panel a groups the models by family, therefore its row order is")
        print("        not the table's. Drawn top to bottom:")
        print("        " + ", ".join(drawn))


def nature_check(fig) -> None:
    """Report what each label measures once the figure is scaled to 180 mm.

    This measures the saved tight bounding box rather than the nominal figsize,
    because the crop changes the scale factor.
    """
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
    ap.add_argument("--tables", default=None,
                    help="directory of T*_body.tex; searched automatically if omitted")
    ap.add_argument("--out", default=".")
    ap.add_argument("--stem", default="fig3_dissociation",
                    help="output basename; the manuscript asks for fig3_dissociation")
    ap.add_argument("--nature", action="store_true",
                    help="draw at 180 mm with Nature type sizes")
    ap.add_argument("--check", action="store_true",
                    help="verify the tables and print the quotable ranges, draw nothing")
    a = ap.parse_args()

    a.tables = find_tables(a.tables)
    cal, mods = read_probe(a.tables)
    sweep = read_sweep(a.tables)

    ok = verify(cal, mods, sweep)
    report_numbers(cal, mods, sweep)
    if a.check:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit(1)

    if a.nature:
        go_nature()

    w = 7.09 if a.nature else 11.6          # 180 mm is Nature's double column
    fig = plt.figure(figsize=(w, w * 8.4 / 11.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.04], height_ratios=[1.10, 1.0],
                          wspace=0.32, hspace=0.42)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    panel_a(ax_a, cal, mods)
    panel_b(ax_b, cal, mods)
    panel_c(ax_c, sweep)
    panel_d(ax_d, sweep)

    head_label(ax_a, "a", "Who acts on whom is recorded, in all eleven", BLUE,
               dx=-0.30, dy=1.135, dxc=-0.245)
    head_label(ax_b, "b", "What is recorded is the order, not the role", ORANGE,
               dx=-0.19, dy=1.135, dxc=-0.135)
    head_label(ax_c, "c", "Recorded at every depth", BLUE,
               dx=-0.19, dy=1.085, dxc=-0.135)
    head_label(ax_d, "d", "Never used by the similarity", ORANGE,
               dx=-0.19, dy=1.085, dxc=-0.135)

    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"{a.stem}.{ext}"), facecolor="white")
    print(f"wrote {a.stem}.png and {a.stem}.pdf in {os.path.abspath(a.out)}")
    nature_check(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()

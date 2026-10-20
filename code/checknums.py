#!/usr/bin/env python3
"""Check every number the manuscript quotes against the generated tables.

The manuscript quotes about sixty numbers.  Re-reading the prose against the tables by
hand is how wrong ones survive, so each quoted value is registered here next to the cell
it came from, and this script reads the regenerated CSVs and reports drift.

    python checknums.py                 # after figdata.py and tables.py
    python checknums.py -v              # also print the cells each check read

Exit status is 1 if any check fails, so it can sit in a pre-commit hook or a Makefile.

A check is one of:
    minmax  the prose quotes a range over a column   ("from 0.158 to 0.889")
    cell    the prose quotes one value               ("0.705 for Qwen3-0.6B")
    ratio   the prose quotes a quotient of two cells ("about 1.6 to 1.7 times")

Claims with no table behind them are listed at the end rather than silently omitted:
those come from run logs or figure data, and they are the ones to be careful with.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

OUT = "results/tables"

# ---------------------------------------------------------------- reading the tables
_cache: dict[str, tuple[list[str], list[list[str]]]] = {}


def table(tid: str):
    """Head and rows of a generated table, found by id whatever its slug."""
    if tid not in _cache:
        hits = glob.glob(os.path.join(OUT, f"{tid}_*.csv"))
        hits = [h for h in hits if not h.endswith("_body.tex")]
        if not hits:
            raise FileNotFoundError(tid)
        with open(sorted(hits)[0], newline="") as fh:
            r = list(csv.reader(fh))
        _cache[tid] = (r[0], r[1:])
    return _cache[tid]


def col(head: list[str], key: str) -> int:
    key = key.lower()
    for i, h in enumerate(head):
        if key in h.lower():
            return i
    raise KeyError(f"no column matching {key!r} in {head}")


NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def nums(s: str) -> list[float]:
    """Every number in a cell.  Handles 0.705, +0.510, 94%, 0.666-0.746, 0.85/0.78, --.

    A dash BETWEEN two digits separates a range; only a leading dash is a minus sign.
    The LaTeX bodies write ranges with "--", so without this every interval came back
    with a negative upper bound."""
    s = str(s).strip()
    if s in ("--", "", "-", "nan"):
        return []
    s = re.sub(r"(?<=\d)\s*(?:--|\u2013|-)\s*(?=\d)", " ", s.replace("%", ""))
    return [float(x) for x in NUM.findall(s)]


def cells(tid: str, colkey: str, rowkey: str | None = None,
          skip: tuple[str, ...] = (), exact: bool = False) -> list[float]:
    """Numbers in one column, optionally restricted to rows matching rowkey.

    exact compares the first cell outright.  Substring matching is a trap here: "role
    swap" is inside "non-role swap", so a substring rowkey silently pulled in the wrong
    condition and widened every range it was asked for."""
    head, rows = table(tid)
    j = col(head, colkey)
    out = []
    for row in rows:
        first, label = row[0].strip().lower(), " ".join(row[:2]).lower()
        if any(s.lower() in label for s in skip):
            continue
        if rowkey:
            k = rowkey.lower()
            if (first != k) if exact else (k not in label):
                continue
        out += nums(row[j])
    return out


# ---------------------------------------------------------------- the registered claims
# (section, what the prose says, kind, args)
CHECKS = [
 # --- R1, the ratio -------------------------------------------------------------------
 ("R1", "ratio runs 0.158 to 0.889",        "minmax", dict(t="T01", c="ratio",
                                                           lo=0.158, hi=0.889)),
 ("R1", "0.442 at Qwen3-0.6B",              "cell",   dict(t="T01", c="ratio",
                                                           r="qwen3-0.6b", v=0.442)),
 ("R1", "0.889 at Qwen3-8B",                "cell",   dict(t="T01", c="ratio",
                                                           r="qwen3-8b", v=0.889)),
 ("R1", "0.158 for the retrained encoder",  "cell",   dict(t="T01", c="ratio",
                                                           r="retrained", v=0.158)),
 ("R1", "eleven models",                    "nrows",  dict(t="T01", n=11)),
 # --- R1, equivalence -----------------------------------------------------------------
 ("R1", "one-sided P at worst 8.9e-08",     "maxof",  dict(t="T02", c="not further",
                                                           v=8.9e-08)),
 ("R1", "ratio column spans 0.705 to 1.095", "minmax", dict(t="T02", c="ratio",
                                                            lo=0.705, hi=1.095)),
 ("R1", "sup-SimCSE ratio 0.869",           "cell",   dict(t="T02", c="ratio",
                                                           r="sup-simcse", v=0.869)),
 ("R1", "unsup-SimCSE ratio 1.095",         "cell",   dict(t="T02", c="ratio",
                                                           r="unsup-simcse", v=1.095)),
 ("R1", "Qwen3-0.6B ratio 0.705",           "cell",   dict(t="T02", c="ratio",
                                                           r="qwen3-0.6b", v=0.705)),
 ("R1", "Qwen3-0.6B interval 0.666-0.746",  "cellpair", dict(t="T02", c="CI",
                                                             r="qwen3-0.6b",
                                                             v=(0.666, 0.746))),
 # --- R2, the probe -------------------------------------------------------------------
 ("R2", "both voices: 61 to 84%",           "minmax", dict(t="T03", c="both",
                                                           lo=60.8, hi=83.8,
                                                           skip=("calibration",))),
 ("R2", "active only, on actives: 68 to 99%",
                                            "minmax", dict(t="T03", c="act",
                                                            lo=68.0, hi=99.2,
                                                            skip=("calibration",))),
 ("R2", "active only, on passives: 1 to 34%",
                                            "minmax", dict(t="T03", c="pas",
                                                            lo=1.1, hi=34.0,
                                                            skip=("calibration",))),
 ("R2", "control, unseen pairs: 51 to 54%", "minmax", dict(t="T03", c="control new",
                                                           lo=50.6, hi=53.7,
                                                           skip=("calibration",))),
 ("R2", "control, familiar pairs: 92 to 98%",
                                            "minmax", dict(t="T03", c="control seen",
                                                            lo=92.4, hi=97.8,
                                                            skip=("calibration",))),
 # --- R3, the mechanism ---------------------------------------------------------------
 ("R3", "role vs role 0.090 to 0.096",      "minmax", dict(t="T09", c="role vs role",
                                                           lo=0.090, hi=0.096)),
 ("R3", "random level 0.025 to 0.030",      "minmax", dict(t="T09", c="random level",
                                                           lo=0.025, hi=0.030)),
 ("R3", "own identity 0.527 to 0.681",      "minmax", dict(t="T09", c="own name",
                                                           lo=0.527, hi=0.681)),
 ("R3", "other identities 0.056 to 0.089",  "minmax", dict(t="T09", c="other names",
                                                           lo=0.056, hi=0.089)),
 ("R3", "unsup ranks 37, 55, 69, 78",       "rowis",  dict(t="T13", r="unsup-simcse",
                                                           v=[37, 55, 69, 78])),
 ("R3", "Qwen3 ranks 23, 41, 56, 68",       "rowis",  dict(t="T13", r="qwen3-0.6b",
                                                           v=[23, 41, 56, 68])),
 ("R3", "at 120 entities, 68 to 78",        "minmax", dict(t="T13", c="120 entities",
                                                           lo=68, hi=78,
                                                           skip=("generic",))),
 ("R3", "role swaps cost 1.6-1.7x, matched rank 64",
                                            "ratio",  dict(t="T10", c="per 0.01",
                                                            a="entity swaps", b="control swaps",
                                                            lo=1.5, hi=1.8, rank="64")),
 # --- R4, the negative ----------------------------------------------------------------
 ("R4", "topic overlap 0.137",              "cell",   dict(t="T07", c="overlap with",
                                                           r="topic", v=0.137)),
 ("R4", "baseline excess +0.510",           "cell",   dict(t="T07", c="overlap excess",
                                                           r="base", v=0.510)),
 ("R4", "topic excess +0.503",              "cell",   dict(t="T07", c="overlap excess",
                                                           r="topic", v=0.503)),
 ("R4", "topic fooled 0.248 to 0.254",      "minmax", dict(t="T07", c="fooled",
                                                           rowkey="topic", exact=True,
                                                           lo=0.248, hi=0.254)),
 ("R4", "shuffled excess +0.502 to +0.511", "minmax", dict(t="T07", c="overlap excess",
                                                           rowkey="shuffled", exact=True,
                                                           lo=0.502, hi=0.511)),
 ("R4", "shuffled fooled 0.260 to 0.266",   "minmax", dict(t="T07", c="fooled",
                                                           rowkey="shuffled", exact=True,
                                                           lo=0.260, hi=0.266)),
 ("R4", "non-role fooled 0.296 to 0.304",   "minmax", dict(t="T07", c="fooled",
                                                           rowkey="non-role swap", exact=True,
                                                           lo=0.296, hi=0.304)),
 ("R4", "role-swap excess +0.317 to +0.337","minmax", dict(t="T07", c="overlap excess",
                                                           rowkey="role swap", exact=True,
                                                           lo=0.317, hi=0.337)),
 ("R4", "role-swap fooled 0.134 to 0.150",  "minmax", dict(t="T07", c="fooled",
                                                           rowkey="role swap", exact=True,
                                                           lo=0.134, hi=0.150)),
 ("R4", "role-swap STS-B 0.726 to 0.729",   "minmax", dict(t="T07", c="STS-B",
                                                           rowkey="role swap", exact=True,
                                                           lo=0.726, hi=0.729)),
 ("R4", "baseline fooled 0.276",            "cell",   dict(t="T07", c="fooled",
                                                           r="base", v=0.276)),
 # --- R4, the cost --------------------------------------------------------------------
 ("R4", "baseline mean nDCG@10 0.384",      "cell",   dict(t="T11", c="mean",
                                                           r="retrained (baseline)", v=0.384)),
 ("R4", "role swap 0.334",                  "cell",   dict(t="T11", c="mean",
                                                           r="role swap", v=0.334)),
 ("R4", "role swap -13.0%",                 "cell",   dict(t="T11", c="change",
                                                           r="role swap", v=-13.0)),
 ("R4", "role swap fooled 0.152",           "cell",   dict(t="T11", c="fooled",
                                                           r="role swap", v=0.152)),
 ("R4", "drift floor -6.0%",                "cell",   dict(t="T11", c="change",
                                                           r="none", v=-6.0)),
 ("R4", "word-shuffled -33.2%",             "cell",   dict(t="T11", c="change",
                                                           r="word-shuffled", v=-33.2)),
 ("R4", "non-role swap -35.6%",             "cell",   dict(t="T11", c="change",
                                                           r="non-role swap", v=-35.6)),
 ("R4", "graded real -8.4%",                "cell",   dict(t="T11", c="change",
                                                           r="graded, real", v=-8.4)),
 ("R4", "graded shuffled -51.1%",           "cell",   dict(t="T11", c="change",
                                                           r="graded, shuffled", v=-51.1)),
 ("R4", "both graded fooled 0.220",         "minmax", dict(t="T11", c="fooled",
                                                           rowkey="graded",
                                                           lo=0.220, hi=0.220)),
 # --- R5, the blind battery -----------------------------------------------------------
 ("R5", "overall AUC spans 0.292 to 0.572", "minmax", dict(t="T14", c="overall",
                                                            lo=0.292, hi=0.572)),
 ("R5", "s/o swap spans 0.036 to 0.489",    "minmax", dict(t="T14", c="subject/object",
                                                            lo=0.036, hi=0.489)),
 ("R5", "E5-base on that subcase 0.489",    "cell",   dict(t="T14", c="subject/object",
                                                           r="e5-base", v=0.489)),
 ("R5", "passive subcase 0.50 to 0.66",     "minmax", dict(t="T14", c="passive",
                                                           lo=0.501, hi=0.656)),
 ("R5", "every interval brackets its point estimate",
                                            "brackets", dict(t="T14", c="overall",
                                                              ci="CI")),
]

# Quoted in the prose with no table behind them.  Not failures: things to keep an eye on.
SOURCELESS = [
 ("R1", "precision moves the ratio by 0.023 and 0.033; size by 0.424, 0.465 to 0.889",
        "T01b, if the crossed run is collected"),
 ("R1", "10/10, 9/10 and 4/10 worked items; factors 3.0, 2.6, 2.2", "T18 / fig_examples"),
 ("R2", "layer ratios 0.010 to 0.184 and 0.009 to 0.443", "T04 / figure 3 data"),
 ("R3", "swap similarity 0.9954 to 0.9844 under a full-rank edit", "e4 intervention log"),
 ("R4", "contrastive loss 1e-4, 1e-3, and 0.18 to 0.26", "training log, no table column"),
 ("R5", "PAWS-X English 0.682 to 0.728", "T15"),
 ("R5", "benchmark excess +0.08 to +0.63", "T16"),
 ("R5", "HANS transfer 0.633-0.781 and 0.651-0.714", "T17"),
]


# ---------------------------------------------------------------- running the checks
def tol(v: float) -> float:
    """Half a unit in the last place the prose quotes, so rounding is not a failure."""
    s = f"{abs(v):.10f}".rstrip("0")
    dp = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10 ** -dp + 1e-12


def run(kind: str, a: dict) -> tuple[bool, str]:
    t = a["t"]
    if kind == "nrows":
        n = len(table(t)[1])
        return n == a["n"], f"{n} rows"
    if kind == "minmax":
        v = cells(t, a["c"], a.get("rowkey"), a.get("skip", ()), a.get("exact", False))
        if not v:
            return False, "no numbers found"
        ok = abs(min(v) - a["lo"]) <= tol(a["lo"]) and abs(max(v) - a["hi"]) <= tol(a["hi"])
        return ok, f"{min(v):g} to {max(v):g}"
    if kind == "maxof":
        v = cells(t, a["c"])
        return (max(v) <= a["v"] * 1.05 if v else False), f"worst {max(v):.2g}" if v else "none"
    if kind == "cell":
        v = cells(t, a["c"], a["r"], exact=a.get("exact", True))
        if not v:
            return False, f"row {a['r']!r} not found"
        return abs(v[0] - a["v"]) <= tol(a["v"]), f"{v[0]:g}"
    if kind == "cellpair":
        v = cells(t, a["c"], a["r"], exact=a.get("exact", True))
        if len(v) < 2:
            return False, f"{v}"
        ok = all(abs(x - y) <= tol(y) for x, y in zip(v[:2], a["v"]))
        return ok, f"{v[0]:g}-{v[1]:g}"
    if kind == "rowis":
        head, rows = table(t)
        for row in rows:
            if a["r"].lower() in row[0].lower():
                got = [x for c in row[1:] for x in nums(c)]
                return got[:len(a["v"])] == [float(x) for x in a["v"]], f"{got}"
        return False, f"row {a['r']!r} not found"
    if kind == "ratio":
        head, rows = table(t)
        j, want = col(head, a["c"]), a.get("rank")
        pick = lambda k: [nums(r[j])[0] for r in rows
                          if k in " ".join(r).lower() and (not want or want in " ".join(r))
                          and nums(r[j])]
        p, q = pick(a["a"]), pick(a["b"])
        if not p or not q:
            return False, "rows not matched"
        v = p[0] / q[0]
        return a["lo"] <= v <= a["hi"], f"{v:.2f}x"
    if kind == "brackets":
        head, rows = table(t)
        jp, jc = col(head, a["c"]), col(head, a["ci"])
        bad = []
        for r in rows:
            p, ci = nums(r[jp]), nums(r[jc])
            if p and len(ci) >= 2 and not (ci[0] <= p[0] <= ci[1]):
                bad.append(f"{r[0]} {p[0]:g} not in {ci[0]:g}-{ci[1]:g}")
        return not bad, ("all bracket" if not bad else "; ".join(bad[:3]))
    raise ValueError(kind)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(OUT):
        print(f"no {OUT}/ -- run: python figdata.py && python tables.py")
        return 1

    fails = missing = 0
    print(f"{'':4s} {'sec':4s} {'claim':52s} {'found':26s}")
    print("-" * 90)
    for sec, claim, kind, a in CHECKS:
        try:
            ok, got = run(kind, a)
        except FileNotFoundError as e:
            print(f"{'SKIP':4s} {sec:4s} {claim[:52]:52s} {f'{e} not generated':26s}")
            missing += 1
            continue
        except (KeyError, IndexError) as e:
            print(f"{'ERR ':4s} {sec:4s} {claim[:52]:52s} {str(e)[:26]:26s}")
            fails += 1
            continue
        if not ok:
            fails += 1
        print(f"{'ok  ' if ok else 'FAIL':4s} {sec:4s} {claim[:52]:52s} {got[:26]:26s}")
        if args.verbose:
            print(f"     {a}")

    print("-" * 90)
    print(f"{len(CHECKS) - fails - missing} ok, {fails} failed, {missing} skipped "
          f"(table not generated)")
    if SOURCELESS:
        print("\nQuoted in the prose with no table column behind it "
              "-- verify these by hand:")
        for sec, claim, where in SOURCELESS:
            print(f"  {sec:4s} {claim}")
            print(f"       -> {where}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

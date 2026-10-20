#!/usr/bin/env python3
"""Compare the generated table bodies against the tables typeset in the .tex files.

This answers one question: does every T*_body.tex still agree, cell for cell and row
for row, with the table that is actually printed in the paper?  It needs nothing but
the .tex sources, so it runs before the PDFs are built and before the bib is tidy.

It separates three different kinds of disagreement, because they mean different things:

  values differ      a number in the paper is not the number the analysis produced.
                     This is the one that matters, and it is a stop-the-submission bug.
  order differs      the same rows, in different positions.  Harmless on its own, yet it
                     means one of the two files was built from an older run, and a
                     reviewer checking "row five" against the figure lands elsewhere.
  rows differ        a row was added or dropped on one side.

Run
    python check_tables_match.py --tables results/tables
    python check_tables_match.py --tables results/tables --dir .
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys


def tabulars(tex: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every tabular in a .tex file, as (header cells, body rows)."""
    out = []
    for m in re.finditer(r"\\begin\{tabular\}.*?\\end\{tabular\}", tex, re.S):
        head, rows = None, []
        for line in m.group(0).splitlines():
            s = line.strip()
            if s.startswith("%") or "&" not in s:
                continue
            s = re.sub(r"\\\\.*$", "", s)
            cells = [c.strip() for c in s.split("&")]
            if head is None:
                head = cells
            else:
                rows.append(cells)
        if head:
            out.append((head, rows))
    return out


def expand(path: str, depth: int = 0, seen: set | None = None) -> str:
    """Read a .tex file with its \\input and \\include files pulled in.

    A paper is often split so the main file holds no tabular at all and every table
    arrives through \\input{tables/T03_body}.  Without following those, the comparison
    silently has nothing to compare, which is worse than failing.
    """
    seen = seen if seen is not None else set()
    path = os.path.abspath(path)
    if path in seen or depth > 6 or not os.path.isfile(path):
        return ""
    seen.add(path)
    tex = open(path, encoding="utf-8", errors="replace").read()
    base = os.path.dirname(path)

    def sub(m):
        name = m.group(1).strip()
        for cand in (name, name + ".tex"):
            p = cand if os.path.isabs(cand) else os.path.join(base, cand)
            if os.path.isfile(p):
                return "\n" + expand(p, depth + 1, seen) + "\n"
        return ""                      # a missing input is the document's problem

    return re.sub(r"\\(?:input|include)\s*\{([^}]*)\}", sub, tex)


def norm(cell: str) -> str:
    """Strip the formatting so a bolded cell still equals its plain twin."""
    c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    c = re.sub(r"\\[a-zA-Z]+", " ", c)
    c = c.replace("$", "").replace("{", "").replace("}", "")
    c = c.replace("--", "-").replace("\u2013", "-")
    return re.sub(r"\s+", "", c).lower()


def key(row: list[str]) -> tuple:
    return tuple(norm(c) for c in row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, help="the folder of T*_body.tex")
    ap.add_argument("--dir", default=".", help="where the .tex sources are")
    ap.add_argument("--docs", default="manuscript.tex,supplementary.tex",
                    help="the documents that are actually submitted, comma separated; "
                         "drafts and backups in the same folder are ignored")
    a = ap.parse_args()

    bodies = sorted(glob.glob(os.path.join(a.tables, "T*_body.tex")))
    if not bodies:
        print(f"no T*_body.tex in {os.path.abspath(a.tables)}")
        return 1

    # Only the two documents that are actually submitted.  A folder like this one
    # accumulates drafts -- results_final.tex, supplementary_v2.tex, _OLD, _backup --
    # and comparing against whichever of those happens to sort first produces a page
    # of alarming nonsense about numbers that no reviewer will ever see.  The first
    # run of this script did exactly that, matching T01 against a stale results.tex.
    want = [w.strip() for w in a.docs.split(",") if w.strip()]
    srcs = [os.path.join(a.dir, w) for w in want]
    missing = [s for s in srcs if not os.path.isfile(s)]
    srcs = [s for s in srcs if os.path.isfile(s)]
    if missing:
        print("not found, so not compared against: "
              + ", ".join(os.path.basename(m) for m in missing))
    if not srcs:
        print(f"none of {', '.join(want)} are in {os.path.abspath(a.dir)}")
        print("Pass the real ones:  --docs mypaper.tex,mysupp.tex")
        return 1
    docs = {os.path.basename(s): tabulars(expand(s)) for s in srcs}
    ntab = sum(len(v) for v in docs.values())
    print(f"{len(bodies)} generated bodies against {ntab} typeset tables in "
          f"{', '.join(docs)}\n")

    value_bugs, order_bugs, row_bugs, unmatched = [], [], [], []
    compared = 0

    for p in bodies:
        tid = os.path.basename(p).split("_")[0]
        tb = tabulars(open(p, encoding="utf-8").read())
        if not tb:
            unmatched.append(f"{tid}: no tabular in the body file")
            continue
        head, rows = tb[0]
        hkey = [norm(c) for c in head]

        hit = None
        for doc, tabs in docs.items():
            for h, r in tabs:
                if [norm(c) for c in h] == hkey:
                    hit = (doc, r)
                    break
            if hit:
                break
        if not hit:
            unmatched.append(f"{tid}: no typeset table with a matching header")
            continue
        doc, typeset = hit

        compared += 1
        gk, tk = [key(r) for r in rows], [key(r) for r in typeset]
        if gk == tk:
            print(f"  {tid:<5} ok            {len(rows)} rows, {doc}")
            continue

        if sorted(gk) == sorted(tk):
            moved = [i for i, (x, y) in enumerate(zip(gk, tk), 1) if x != y]
            order_bugs.append(
                f"{tid} ({doc}): same {len(rows)} rows, {len(moved)} in different "
                f"positions (rows {', '.join(map(str, moved))})")
            print(f"  {tid:<5} ORDER         {len(moved)} rows sit elsewhere, {doc}")
            continue

        only_gen = [r for r in gk if r not in tk]
        only_typ = [r for r in tk if r not in gk]
        if len(rows) != len(typeset):
            row_bugs.append(f"{tid} ({doc}): {len(rows)} rows generated, "
                            f"{len(typeset)} typeset")
            print(f"  {tid:<5} ROW COUNT     {len(rows)} vs {len(typeset)}, {doc}")
            continue

        # same count, same labels, different cells: a number moved
        for i, (g, t) in enumerate(zip(rows, typeset), 1):
            if key(g) != key(t):
                diff = [(h, x, y) for h, x, y in zip(head, g, t) if norm(x) != norm(y)]
                for h, x, y in diff:
                    value_bugs.append(
                        f"{tid} ({doc}) row {i} [{g[0]}] column '{h}': "
                        f"generated {x!r}, typeset {y!r}")
        print(f"  {tid:<5} VALUES        {len(only_gen)} rows disagree, {doc}")

    print()
    if value_bugs:
        print("VALUES DIFFER -- a number in the paper is not what the analysis produced")
        for b in value_bugs[:40]:
            print("   " + b)
        if len(value_bugs) > 40:
            print(f"   ... and {len(value_bugs) - 40} more")
        print()
    if row_bugs:
        print("ROWS DIFFER")
        for b in row_bugs:
            print("   " + b)
        print()
    if order_bugs:
        print("ORDER DIFFERS -- same rows, different positions.  The values are fine,")
        print("therefore nothing is wrong in the paper, yet one of the two files was")
        print("built from an older run and the two should be regenerated together.")
        for b in order_bugs:
            print("   " + b)
        print()
    if unmatched:
        print("NOT COMPARED -- no typeset table had a matching header.  This is")
        print("expected where the typeset version drops a column the body keeps, as")
        print("T18 does with its sentences; verify_paper.py compares those on their")
        print("shared columns.  Any other table listed here is worth a look.")
        for b in unmatched:
            print("   " + b)
        print()

    if compared == 0:
        print("NOTHING WAS COMPARED, therefore this run proves nothing.")
        print("Every generated body failed to find a typeset table with the same")
        print("header, which almost always means the documents named by --docs are")
        print("not the ones holding the tables.  Find the real sources with:")
        print("   grep -rl 'begin{tabular}' --include='*.tex' .")
        print("then point the check at them:")
        print("   python check_tables_match.py --tables <dir> --docs <main>,<supp>")
        return 1
    if value_bugs or row_bugs:
        print("FAIL: the paper and the analysis disagree on a value.")
        return 1
    if order_bugs:
        print("PASS on values, with an ordering difference to tidy up.")
        return 0
    print(f"PASS: all {compared} compared tables match their typeset twin, row for "
          f"row.")
    if len(unmatched) > compared:
        print(f"      Note that {len(unmatched)} bodies were not compared at all, "
              f"which is more\n      than were, so read that list above before "
              f"trusting this line.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Audit a tables folder that has grown two naming conventions.

A folder can end up holding both T03_body.tex and T03_a_long_descriptive_name.tex.
The figure scripts read one of them.  If the paper is built from the other, and the
two have drifted, the figures and the tables quietly stop agreeing, and nothing in
the build says so.

This reports, for every table id, which files claim it and whether their tabulars are
identical.  It also finds the .tex documents outside the folder that actually contain
a tabular, which is how you locate the real manuscript and supplementary when they
are not where you expected.

Run
    python audit_tables_folder.py --tables results/tables
    python audit_tables_folder.py --tables results/tables --root ~/CHVLM
"""
from __future__ import annotations

import argparse
import os
import re
import sys

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".ipynb_checkpoints", "venv",
             ".venv", "env", "site-packages", ".cache", "wandb", "checkpoints"}


def tabular(tex: str):
    """The first tabular in a file, as (header cells, body rows), or None."""
    m = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", tex, re.S)
    if not m:
        return None
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
    return (head, rows) if head else None


def norm(cell: str) -> str:
    c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    c = re.sub(r"\\[a-zA-Z]+", " ", c)
    c = c.replace("$", "").replace("{", "").replace("}", "")
    c = c.replace("--", "-").replace("\u2013", "-")
    return re.sub(r"\s+", "", c).lower()


def sig(tab):
    head, rows = tab
    return ([norm(c) for c in head], [[norm(c) for c in r] for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--root", default=None,
                    help="where to look for the paper sources; defaults to the "
                         "parent of the tables folder")
    a = ap.parse_args()

    tdir = os.path.abspath(os.path.expanduser(a.tables))
    if not os.path.isdir(tdir):
        print(f"no such folder: {tdir}")
        return 1
    files = sorted(os.listdir(tdir))
    print(f"{len(files)} files in {tdir}\n")

    # group by the table id at the front of the name
    groups: dict[str, list[str]] = {}
    other: list[str] = []
    for f in files:
        m = re.match(r"(T\d+[a-z]?)[_.]", f)
        if m and f.endswith(".tex"):
            groups.setdefault(m.group(1), []).append(f)
        elif f.endswith(".tex"):
            other.append(f)

    def idnum(k):
        m = re.match(r"T(\d+)([a-z]?)", k)
        return (int(m.group(1)), m.group(2))

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"{len(groups)} table ids, {len(multi)} of them claimed by more than one "
          f"file\n")

    drifted, agreed, unreadable = [], [], []
    for k in sorted(multi, key=idnum):
        names = sorted(multi[k], key=lambda n: (0 if n.endswith("_body.tex") else 1, n))
        tabs = {}
        for n in names:
            t = tabular(open(os.path.join(tdir, n), encoding="utf-8",
                             errors="replace").read())
            tabs[n] = t
        ref = names[0]
        if tabs[ref] is None:
            unreadable.append(f"{k}: {ref} has no tabular")
            continue
        rows_ref = len(tabs[ref][1])
        print(f"  {k}")
        same_all = True
        for n in names:
            t = tabs[n]
            if t is None:
                print(f"       {n:<56} no tabular")
                same_all = False
                continue
            mark = ""
            if n != ref:
                if sig(t) == sig(tabs[ref]):
                    mark = "same as the _body file"
                else:
                    same_all = False
                    hs, rs = sig(t)
                    hr, rr = sig(tabs[ref])
                    if hs != hr:
                        mark = "DIFFERENT COLUMNS"
                    elif len(rs) != len(rr):
                        mark = f"DIFFERENT ROW COUNT ({len(rs)} vs {len(rr)})"
                    else:
                        bad = [i for i, (x, y) in enumerate(zip(rs, rr), 1) if x != y]
                        mark = f"DIFFERENT VALUES at row {bad[0]}" if bad else "differs"
            print(f"       {n:<56} {len(t[1]):>3} rows   {mark}")
        (agreed if same_all else drifted).append(k)
    if multi:
        print()

    if other:
        print(f"{len(other)} .tex files in the folder with no table id in the name:")
        for f in other[:12]:
            print(f"   {f}")
        if len(other) > 12:
            print(f"   ... and {len(other) - 12} more")
        print()

    # where are the paper sources?
    root = os.path.abspath(os.path.expanduser(a.root or os.path.dirname(tdir)))
    hits = []
    for cur, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        if os.path.abspath(cur) == tdir:
            continue
        for f in fs:
            if f.endswith(".tex"):
                p = os.path.join(cur, f)
                try:
                    tex = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                n = len(re.findall(r"\\begin\{tabular\}", tex))
                ninp = len(re.findall(r"\\(?:input|include)\s*\{", tex))
                if n or ninp:
                    hits.append((os.path.relpath(p, root), n, ninp))
    print(f"documents under {root}, outside the tables folder, that hold tables:")
    if not hits:
        print("   none -- no .tex out here contains a tabular or an \\input,")
        print("   therefore the manuscript and supplementary are not in this tree.")
    else:
        hits.sort(key=lambda h: (-h[1] - h[2], h[0]))
        print(f"   {'file':<52}{'tabulars':>9}{'inputs':>8}")
        for p, n, ni in hits[:25]:
            print(f"   {p:<52}{n:>9}{ni:>8}")
        if len(hits) > 25:
            print(f"   ... and {len(hits) - 25} more")
    print()

    if drifted:
        print("THE TWO COPIES DISAGREE for: " + ", ".join(sorted(drifted, key=idnum)))
        print("Decide which naming convention is current and delete the other, because")
        print("the figure scripts read T*_body.tex and the paper may be built from the")
        print("descriptive files.")
        return 1
    if multi:
        print(f"Every duplicated id agrees across its copies ({len(agreed)} checked), "
              f"so the\nduplication is untidy rather than dangerous.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

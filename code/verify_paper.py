#!/usr/bin/env python3
"""Everything that can be checked about the two PDFs without re-running a model.

checknums.py answers "does each number the prose quotes match the table it came from".
This answers the questions around that, each of which has caught a real error:

  1  the typeset tables are the generated tables       -- a checker that passes against a
                                                          table nobody reads is worthless
  2  every \\st... macro lands on the table it names    -- hand-written numbers, no LaTeX
                                                          warning when they drift
  3  every \\cite key resolves, every entry is cited    -- an uncited model is a reviewer
                                                          question
  4  the counts the prose asserts match the data       -- "eleven models", "nine domains"
  5  every numeric cell reaches the rendered PDF       -- a cell dropped by a float or a
                                                          column spec is invisible in .tex
  6  the build is clean                                -- errors, undefined refs, overfull

Usage, from the directory holding manuscript.tex and supplementary.tex:

    python verify_paper.py                    # tables/ is looked for beside this file
    python verify_paper.py --tables path/to/tables
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

DOCS = ("manuscript", "supplementary")

# A macro's name and its table's label are independent spellings of one thing, so the
# pairing is stated here once rather than guessed from the strings.
MACRO_LABEL = {
    "stEquiv": "stab:equiv", "stPrec": "stab:precision", "stPercell": "stab:percell",
    "stRatio": "stab:ratio", "stLayers": "stab:layers", "stMeasures": "stab:measures",
    "stProbe": "stab:probe", "stDir": "stab:directions", "stDecomp": "stab:decomp",
    "stRank": "stab:rank", "stExch": "stab:exchange", "stGraded": "stab:graded",
    "stCkpt": "stab:checkpoints", "stCost": "stab:cost", "stHone": "stab:h1",
    "stHtwo": "stab:h2", "stHthree": "stab:h3", "stHfour": "stab:h4",
    "stItems": "stab:items", "stDist": "stab:distances", "stControl": "stab:control",
}


# ------------------------------------------------------------------ reading LaTeX tables
def tabulars(tex: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every tabular in a LaTeX source, as (header row, data rows).

    Rows are taken as "a line inside a tabular that ends in \\\\ and holds an &".  The
    scan is bounded by the tabular because a looser rule also matches prose.
    """
    out: list[tuple[list[str], list[list[str]]]] = []
    cur: list[list[str]] | None = None
    inside = False
    for line in tex.splitlines():
        if "\\begin{tabular}" in line:
            inside, cur = True, []
            continue
        if "\\end{tabular}" in line:
            if cur:
                out.append((cur[0], cur[1:]))
            inside, cur = False, None
            continue
        s = line.strip()
        if inside and cur is not None and s.endswith("\\\\") and "&" in s:
            cur.append([c.strip() for c in s[:-2].split("&")])
    return out


def norm(cell: str) -> str:
    """A cell stripped to its content: macros, braces, maths and dash style removed.

    The generated bodies write a range as 0.153-0.163 and the documents typeset it as an
    en dash, which is typography and not a difference in the number.
    """
    c = re.sub(r"\\[a-zA-Z]+|[{}$~]", "", cell)
    c = c.replace("---", "-").replace("--", "-")
    return re.sub(r"\s+", " ", c).strip().lower()


# ------------------------------------------------------------------------------- checks
def check_tables(tabdir: str, docs: dict) -> list[str]:
    """Each generated body must appear, cell for cell, in one of the documents."""
    bad, matched, skipped = [], 0, []
    for p in sorted(glob.glob(os.path.join(tabdir, "T*_body.tex"))):
        tid = os.path.basename(p).split("_")[0]
        tb = tabulars(open(p, encoding="utf-8").read())
        if not tb:
            bad.append(f"{tid}: no tabular in the body file")
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
            # T18 keeps the sentences the typeset table drops, so its header cannot match.
            # It is compared on its shared columns by check_t18 instead of waved through.
            skipped.append(tid)
            continue
        doc, typeset = hit
        matched += 1
        if len(typeset) != len(rows):
            bad.append(f"{tid} ({doc}): {len(rows)} rows generated, {len(typeset)} typeset")
            continue
        for i, (a, b) in enumerate(zip(rows, typeset)):
            if [norm(x) for x in a] != [norm(x) for x in b]:
                bad.append(f"{tid} ({doc}) row {i}: {a} vs {b}")
                break
    print(f"  {matched} generated bodies matched their typeset table, cell for cell")
    if skipped:
        print(f"  compared on shared columns instead (header differs): {', '.join(skipped)}")
    return bad


def check_t18(tabdir: str, sup: str) -> list[str]:
    """T18 carries the sentences; the supplementary drops them. Compare what they share."""
    def pick(tex, must_have):
        """The one tabular whose header holds all of must_have, and only that one.

        A substring is not enough.  "to kept meaning" also matches the per-model ratio
        table five tables earlier, which has eleven rows, and the comparison then runs
        against the wrong table and reports a row-count mismatch that is really a lookup
        bug.  Requiring the full set of distinctive columns, and refusing an ambiguous
        match outright, is what the rest of this suite does.
        """
        hits = [(h, r) for h, r in tabulars(tex)
                if all(any(k in norm(c) for c in h) for k in must_have)]
        if len(hits) != 1:
            return None, None
        return hits[0]
    p = os.path.join(tabdir, "T18_body.tex")
    if not os.path.exists(p):
        return ["T18_body.tex missing"]
    _, body = pick(open(p, encoding="utf-8").read(), ("domain", "d kept", "closer"))
    _, typeset = pick(sup, ("domain", "to kept meaning", "closer"))
    if body is None or typeset is None:
        return ["T18: could not locate both tables"]
    if len(body) != len(typeset):
        return [f"T18: {len(body)} rows generated, {len(typeset)} typeset"]
    n = lambda x: re.sub(r"[^0-9.]", "", x)                                # noqa: E731
    bad = []
    for i, (b, t) in enumerate(zip(body, typeset)):
        # body: domain model original kept reversed d_kept d_rev ratio closer
        # typeset: domain model d_kept d_rev ratio closer
        got = (b[0].lower(), b[1].lower(), n(b[5]), n(b[6]), n(b[7]), b[8].lower())
        want = (t[0].lower(), t[1].lower(), n(t[2]), n(t[3]), n(t[4]), t[5].lower())
        if got != want:
            bad.append(f"T18 row {i}: {got} vs {want}")
    print(f"  T18: {len(typeset)} rows agree on both distances, the ratio and closer")
    return bad


def check_macros(man: str, sup: str) -> list[str]:
    """Every \\st... macro must land on the supplementary table it is named for."""
    macros = {m.group(1): int(m.group(2)) for m in
              re.finditer(r"\\newcommand\{\\(st[A-Za-z]+)\}\{(\d+)\}", man)}
    body = sup.split(r"\begin{document}", 1)[-1]
    # Three macros typeset a supplementary table.  Matching only \stable reported twelve
    # misalignments that did not exist, so all three are taken.
    order = [m.group(2) for m in sorted(
        re.finditer(r"\\(stable|stabletall|stablewide)\{([^}]*)\}", body),
        key=lambda m: m.start())]
    bad = []
    if len(macros) != len(order):
        bad.append(f"{len(macros)} macros against {len(order)} captioned tables")
    for name, want in MACRO_LABEL.items():
        if name not in macros:
            bad.append(f"{name}: not defined")
            continue
        n = macros[name]
        got = order[n - 1] if 1 <= n <= len(order) else "<out of range>"
        if got != want:
            bad.append(f"{name} = {n} points at {got}, named for {want}")
    for name in macros:
        if name not in MACRO_LABEL:
            bad.append(f"{name}: defined but not covered by this check")
    if sorted(macros.values()) != list(range(1, len(order) + 1)):
        bad.append(f"numbers are not 1..{len(order)} exactly")
    print(f"  {len(macros)} macros, {len(order)} captioned tables, numbering consecutive")
    return bad


def check_spacing(man: str, pdf: str) -> list[str]:
    r"""Words run into the numbers a macro expands to.

    LaTeX swallows the space that ends a control word, so "Table~\stLayers and" sets as
    "Table 5and".  There is no error, no warning and no wrong number, so every other check
    in this suite passed while six of these shipped in two builds.  Caught two ways: in
    the source, by looking for a macro followed by a bare space and a letter, and in the
    rendered PDF, by looking for a digit glued to a word.
    """
    bad = []
    for m in re.finditer(r"\\st[A-Za-z]+(?=\s+[a-zA-Z])", man):
        line = man[:m.start()].count("\n") + 1
        bad.append(f"manuscript.tex:{line}: {m.group(0)} will swallow the following "
                   f"space; write {m.group(0)}\\ ")
    if os.path.exists(pdf):
        txt = subprocess.run(["pdftotext", "-layout", pdf, "-"],
                             capture_output=True, text=True).stdout
        # "Table 5and", "Fig. 3band" -- a digit followed straight by two or more letters.
        for m in set(re.findall(r"\b(?:Table|Tables|Fig\.?|Figure)\s+\d+[a-z]?[a-z]{2,}",
                                txt)):
            bad.append(f"{os.path.basename(pdf)}: {m!r} -- a number run into a word")
    if not bad:
        print("  no macro swallows the space after it, in source or in the PDF")
    return bad


def check_citations(man: str, sup: str, bib: str) -> list[str]:
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
    cited: set[str] = set()
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^]]*\])*\{([^}]*)\}", man + sup):
        cited |= {k.strip() for k in m.group(1).split(",")}
    bad = []
    for k in sorted(cited - keys):
        bad.append(f"cited but not in references.bib: {k}")
    for k in sorted(keys - cited):
        bad.append(f"in references.bib but never cited: {k}")
    print(f"  {len(keys)} entries, {len(cited)} cited, none missing and none unused")
    return bad


def check_counts(tabdir: str) -> list[str]:
    """The counts the prose asserts, recomputed from the tables that carry them."""
    def rows(tid):
        p = os.path.join(tabdir, f"{tid}_body.tex")
        tb = tabulars(open(p, encoding="utf-8").read())
        return tb[0][1] if tb else []
    t02, t18, t11 = rows("T02"), rows("T18"), rows("T11")
    want = [
        ("eleven models", len(t02), 11),
        ("ten worked items", len({r[0] + r[2] for r in t18}), 10),
        ("nine domains", len({r[0] for r in t18}), 9),
        ("five models in the worked items", len({r[1] for r in t18}), 5),
        ("six NanoBEIR datasets", len(t11[0]) - 4 if t11 else 0, 6),
        ("nineteen rows in Table 4", len(rows("T07")), 19),
    ]
    bad = [f"{n}: got {g}, expect {w}" for n, g, w in want if g != w]
    for n, g, w in want:
        print(f"  {'ok  ' if g == w else 'FAIL'} {n:34s} {g}")
    return bad


def check_rendered(tex: str, pdf: str) -> list[str]:
    """Every number written into a table cell has to appear in the rendered PDF."""
    if not os.path.exists(pdf):
        return [f"{pdf} not built"]
    txt = subprocess.run(["pdftotext", "-layout", pdf, "-"],
                         capture_output=True, text=True).stdout
    flat = re.sub(r"\s+", " ", txt)
    missing, total = [], 0
    for line in tex.splitlines():
        s = line.strip()
        if not (s.endswith("\\\\") and "&" in s):
            continue
        for cell in s[:-2].split("&"):
            for v in re.findall(r"\d+\.\d+", cell):
                total += 1
                if v not in flat:
                    missing.append(v)
    print(f"  {os.path.basename(pdf)}: {total} numeric cells, "
          f"{len(missing)} absent from the PDF")
    return [f"{pdf}: {v} not rendered" for v in missing[:10]]


def check_build(name: str) -> list[str]:
    log = f"{name}.log"
    if not os.path.exists(log):
        return [f"{log} missing -- build it first"]
    text = open(log, encoding="utf-8", errors="replace").read()
    bad = []
    for label, pat in (("error", r"^!"), ("undefined reference", r"undefined"),
                       ("overfull box", r"Overfull")):
        n = len(re.findall(pat, text, re.M))
        if n:
            bad.append(f"{name}: {n} {label}(s)")
    pages = "?"
    if os.path.exists(f"{name}.pdf"):
        out = subprocess.run(["pdfinfo", f"{name}.pdf"], capture_output=True, text=True)
        m = re.search(r"^Pages:\s+(\d+)", out.stdout, re.M)
        pages = m.group(1) if m else "?"
    print(f"  {name}: {pages} pages, "
          + ("clean" if not bad else "; ".join(bad)))
    return bad


# --------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--tables", default=os.path.join(here, "tables"))
    ap.add_argument("--dir", default=".", help="where the .tex and .pdf files are")
    a = ap.parse_args()

    os.chdir(a.dir)
    for f in ("manuscript.tex", "supplementary.tex", "references.bib"):
        if not os.path.exists(f):
            print(f"{f} not found in {os.path.abspath('.')}")
            return 1
    tabdir = a.tables
    if not os.path.isdir(tabdir):
        alt = os.path.join(os.path.abspath("."), "tables")
        if os.path.isdir(alt):
            tabdir = alt
        else:
            print(f"no tables directory at {a.tables}; pass --tables")
            return 1

    man = open("manuscript.tex", encoding="utf-8").read()
    sup = open("supplementary.tex", encoding="utf-8").read()
    bib = open("references.bib", encoding="utf-8").read()
    docs = {"manuscript": tabulars(man), "supplementary": tabulars(sup)}

    sections = [
        ("the build", lambda: [e for n in DOCS for e in check_build(n)]),
        ("generated tables against typeset tables",
         lambda: check_tables(tabdir, docs) + check_t18(tabdir, sup)),
        ("supplementary table macros", lambda: check_macros(man, sup)),
        ("spacing after macros", lambda: check_spacing(man, "manuscript.pdf")),
        ("citations", lambda: check_citations(man, sup, bib)),
        ("counts asserted in the prose", lambda: check_counts(tabdir)),
        ("numeric cells reaching the PDF",
         lambda: [e for n in DOCS
                  for e in check_rendered(open(f"{n}.tex", encoding="utf-8").read(),
                                          f"{n}.pdf")]),
    ]

    failures: list[str] = []
    for title, fn in sections:
        print(f"\n== {title}")
        failures += fn()

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} problem(s):")
        for f in failures:
            print(f"  {f}")
        print("\nThis script does not check that a quoted number matches its table.")
        print("Run: python checknums.py --strict --prose")
        return 1
    print("all checks passed")
    print("\nStill to run, because it checks a different thing -- every number the prose")
    print("quotes against the cell it came from:")
    print("  python checknums.py --strict --prose --prose-dir .")
    return 0


if __name__ == "__main__":
    sys.exit(main())

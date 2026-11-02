#!/usr/bin/env python3
"""Build the real-text role-reversal set from SemEval-2010 Task 8.

Why this corpus. Its sentences are real, its two entities are marked inline, and
its nineteen labels are nine relation types in BOTH directions plus Other. So
exchanging the two spans provably changes the class, by the corpus's own
annotation rather than by our judgement, and the rate at which each direction
occurs is measurable rather than assumed.

Two numbers come out of it:

  base rate    the share of relation instances running in each direction, which
               is the rate at which a corpus of this text carries a claim and
               its reversal as distinct, separately attested statements

  items        a real sentence, its direction-swapped counterpart, and the label
               each one asserts, ready for the retrieval and ratio measurements

Usage
-----
    python semeval_probe.py                       # writes results/semeval_items.jsonl
    python semeval_probe.py --limit 200           # quick look first
"""

from __future__ import annotations

import argparse, collections, json, os, re, sys

# The six relation types that express an action or a directed transfer between
# two participants, so that exchanging them reverses the claim. The three left
# out, Component-Whole, Content-Container and Member-Collection, are part-whole
# and containment relations whose "X of Y" surface form turns a swap into a type
# violation rather than a reversed statement.
DIRECTED = ["Cause-Effect", "Entity-Destination", "Entity-Origin",
            "Instrument-Agency", "Message-Topic", "Product-Producer"]

E1 = re.compile(r"<e1>(.*?)</e1>", re.S)
E2 = re.compile(r"<e2>(.*?)</e2>", re.S)


def strip_tags(s: str) -> str:
    return re.sub(r"</?e[12]>", "", s)


def fix_articles(s: str) -> str:
    """Repair a/an agreement broken by the exchange, so the reversal is not
    identifiable as the odd one out by a grammatical slip."""
    s = re.sub(r"\ba(?=\s+[aeiouAEIOU])", "an", s)
    s = re.sub(r"\bA(?=\s+[aeiouAEIOU])", "An", s)
    s = re.sub(r"\ban(?=\s+[^aeiouAEIOU\W\d_])", "a", s)
    s = re.sub(r"\bAn(?=\s+[^aeiouAEIOU\W\d_])", "A", s)
    return s


def swap_entities(tagged: str) -> str | None:
    """Exchange the two marked spans, leaving every other word in place.

    The tags mark position, so the swap moves the entities and nothing else,
    which is the real-text analogue of the name transposition in the twins."""
    m1, m2 = E1.search(tagged), E2.search(tagged)
    if not m1 or not m2:
        return None
    a, b = m1.group(1), m2.group(1)
    if a.strip().lower() == b.strip().lower():
        return None                                   # a swap that changes nothing
    out = tagged[:m1.start()] + f"<e1>{b}</e1>" + tagged[m1.end():]
    m2b = E2.search(out)
    out = out[:m2b.start()] + f"<e2>{a}</e2>" + out[m2b.end():]
    return out


def flip(label: str) -> str:
    """Cause-Effect(e1,e2) <-> Cause-Effect(e2,e1)."""
    if label.endswith("(e1,e2)"):
        return label[:-7] + "(e2,e1)"
    if label.endswith("(e2,e1)"):
        return label[:-7] + "(e1,e2)"
    return label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/semeval_items.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all-types", action="store_true",
                    help="keep every relation type, not only the directed six")
    a = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    try:
        ds = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8")
    except Exception as e1:
        try:
            ds = load_dataset("sem_eval_2010_task_8")
        except Exception as e2:
            sys.exit(f"neither mirror loaded:\n  {e1}\n  {e2}")

    names = ds["train"].features["relation"].names
    rows = [r for split in ds for r in ds[split]]
    if a.limit:
        rows = rows[:a.limit]
    print(f"[loaded] {len(rows)} sentences, {len(names)} labels")

    # ---- base rate: how often each direction is stated -------------------
    per_type = collections.defaultdict(collections.Counter)
    for r in rows:
        lab = names[r["relation"]]
        if lab == "Other":
            continue
        kind, direction = lab[:-7], lab[-7:]
        per_type[kind][direction] += 1

    print("\n[base rate] both directions are attested in real text\n")
    print(f"  {'relation':<22}{'(e1,e2)':>9}{'(e2,e1)':>9}{'minority':>10}")
    tot = mino = 0
    for kind in sorted(per_type):
        c = per_type[kind]
        f, b = c["(e1,e2)"], c["(e2,e1)"]
        n, m = f + b, min(f, b)
        tot, mino = tot + n, mino + m
        print(f"  {kind:<22}{f:>9}{b:>9}{m / n:>9.1%}")
    print(f"  {'all':<22}{'':>9}{'':>9}{mino / tot:>9.1%}")
    print(f"\n  {mino} of {tot} directed instances state the minority direction of "
          f"their relation,\n  so a corpus of this text holds claims in both "
          f"directions at a measurable rate.")

    # ---- items: a real sentence and its direction-swapped counterpart ----
    items, skipped = [], 0
    for i, r in enumerate(rows):
        lab = names[r["relation"]]
        if lab == "Other":
            continue
        if not a.all_types and lab[:-7] not in DIRECTED:
            skipped += 1
            continue
        sw = swap_entities(r["sentence"])
        if sw is None:
            skipped += 1
            continue
        items.append(dict(id=i, label=lab, swapped_label=flip(lab),
                          sentence=fix_articles(strip_tags(r["sentence"])),
                          reversed=fix_articles(strip_tags(sw)),
                          e1=E1.search(r["sentence"]).group(1),
                          e2=E2.search(r["sentence"]).group(1)))

    kept = "every type" if a.all_types else ", ".join(DIRECTED)
    print(f"\n[items] {len(items)} built, {skipped} skipped")
    print(f"        relation types kept: {kept}")
    if items:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"        wrote {a.out}\n")
        it = items[0]
        print(f"  label     {it['label']}  ->  {it['swapped_label']}")
        print(f"  sentence  {it['sentence'][:160]}")
        print(f"  reversed  {it['reversed'][:160]}")


if __name__ == "__main__":
    main()

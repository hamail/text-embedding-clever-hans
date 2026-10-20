"""
fig_examples.py -- real, measured examples for the figures.

Each item is a triple: an original sentence, a rewording that KEEPS the meaning while
changing the words, and a rewording that REVERSES who acts on whom while keeping every
word. For each model it reports the two distances and which of the two the similarity
places closer. Items are drawn from several domains so the figure is not carried by one
template, and a few are included where the models are expected to do better, so that the
figure can show successes alongside failures.

Nothing here is tuned: the items were written before they were run, and every one is
reported in the output, whichever way it came out.

Usage (from ~/CHVLM/revised):
  python fig_examples.py
  python fig_examples.py --models unsup-simcse bge-base qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import chvlm

OUT = "results/figs"

# (domain, original, same meaning / different words, opposite meaning / same words)
ITEMS = [
    ("pharmacology",
     "Ritonavir inhibits the metabolism of midazolam.",
     "The metabolism of midazolam is slowed by ritonavir.",
     "Midazolam inhibits the metabolism of ritonavir."),
    ("pharmacology",
     "Rifampicin induces the enzyme that clears warfarin.",
     "Warfarin is cleared faster because rifampicin induces the enzyme.",
     "Warfarin induces the enzyme that clears rifampicin."),
    ("epidemiology",
     "The virus was transmitted from the patient to the nurse.",
     "The nurse caught the virus from the patient.",
     "The virus was transmitted from the nurse to the patient."),
    ("molecular biology",
     "The kinase phosphorylates the receptor.",
     "The receptor is phosphorylated by the kinase.",
     "The receptor phosphorylates the kinase."),
    ("law",
     "The plaintiff sued the defendant for breach of contract.",
     "The defendant was sued by the plaintiff for breach of contract.",
     "The defendant sued the plaintiff for breach of contract."),
    ("clinical",
     "The donor's kidney was transplanted into the recipient.",
     "The recipient received a kidney from the donor.",
     "The recipient's kidney was transplanted into the donor."),
    ("news",
     "Germany defeated Brazil in the semi-final.",
     "Brazil lost to Germany in the semi-final.",
     "Brazil defeated Germany in the semi-final."),
    ("finance",
     "Barclays acquired Lehman's brokerage arm.",
     "Lehman's brokerage arm was bought by Barclays.",
     "Lehman acquired Barclays' brokerage arm."),
    # two where the reversal is not merely a reordering, as a contrast
    ("negation",
     "The trial showed the drug reduces mortality.",
     "Mortality fell in the group given the drug.",
     "The trial showed the drug does not reduce mortality."),
    ("quantity",
     "The dose was increased from 5 mg to 20 mg.",
     "The dose went up to 20 mg from 5 mg.",
     "The dose was increased from 20 mg to 5 mg."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["unsup-simcse", "bge-base", "e5-base", "qwen3-0.6b", "qwen3-8b"],
                    choices=list(chvlm.MODELS))
    args = ap.parse_args()

    texts = [t for it in ITEMS for t in it[1:]]
    result = {}
    print(f"{len(ITEMS)} items, {len(args.models)} models. d = 1 - centred cosine.\n")
    for name in args.models:
        m = chvlm.load(name, eager=False)
        E = chvlm.embed(m, texts)
        del m
        mu = E.mean(0)
        rows = []
        print(f"=== {name}")
        print(f"{'domain':16s} {'d(paraphrase)':>13s} {'d(reversal)':>12s} {'ratio':>7s}  closer")
        for k, (dom, orig, kept, rev) in enumerate(ITEMS):
            o, kp, rv = E[3 * k], E[3 * k + 1], E[3 * k + 2]
            dk = float(1 - chvlm.cosine(o[None], kp[None], mu)[0])
            dr = float(1 - chvlm.cosine(o[None], rv[None], mu)[0])
            closer = "reversal" if dr < dk else "paraphrase"
            rows.append(dict(domain=dom, original=orig, kept=kept, reversed=rev,
                             d_kept=dk, d_reversed=dr, ratio=dr / dk, closer=closer))
            print(f"{dom:16s} {dk:13.4f} {dr:12.4f} {dr / dk:7.2f}  "
                  f"{'REVERSAL' if closer == 'reversal' else 'paraphrase'}")
        n_bad = sum(r["closer"] == "reversal" for r in rows)
        print(f"  -> the reversal is closer on {n_bad} of {len(rows)} items\n")
        result[name] = rows

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "examples.json")
    json.dump(dict(items=[dict(zip(("domain", "original", "kept", "reversed"), it))
                          for it in ITEMS], results=result), open(path, "w"), indent=2)
    print(f"[done] wrote {path}")
    print("Report every item, including those where the paraphrase wins: a figure that "
          "shows only failures is not evidence about a model, only about a selection.")


if __name__ == "__main__":
    main()

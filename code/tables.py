#!/usr/bin/env python
"""tables.py -- every table in the paper, built from results/figs/figdata.json.

Same rule as the figures: nothing is typed in. Each table is assembled from the JSON an
experiment wrote, so a table cannot drift from its run. A table whose source has not been
collected is skipped and named in the report rather than filled in.

  python figdata.py        first -- it is what this reads
  python tables.py         write every table
  python tables.py --only T01 T12
  python tables.py --check WRITING_NOTES.md   compare each number against the notes

Writes, for each table:
  results/tables/<id>_<slug>.csv    to check and to re-sort
  results/tables/<id>_<slug>.tex    to paste into the manuscript
  results/tables/<id>_body.tex      the tabular alone, to \\input into the manuscript
  results/tables/ALL_TABLES.md      all of them in one file, for the working notes
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

IN = "results/figs/figdata.json"
IN_DEMO = "results/figs/figdata_demo.json"
OUT = "results/tables"

NICE = {"unsup-simcse": "unsup-SimCSE", "sup-simcse": "sup-SimCSE", "bge-base": "BGE-base",
        "e5-base": "E5-base", "gte-base": "GTE-base", "mpnet-base": "MPNet-base",
        "mxbai-large": "mxbai-large", "retrained": "retrained", "graded": "graded",
        "qwen3-0.6b": "Qwen3-0.6B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B"}

# Different runs were launched with different spellings of the same checkpoint, so one
# model arrives under both a short name and a full HF id.  Without folding these, T05
# shows one model as two rows carrying two runs' numbers, which reads as two models.
ALIAS = {"unsup-simcse-bert-base-uncased": "unsup-simcse",
         "sup-simcse-bert-base-uncased": "sup-simcse",
         "bge-base-en-v1.5": "bge-base",
         "e5-base-v2": "e5-base",
         "gte-base-en-v1.5": "gte-base",
         "all-mpnet-base-v2": "mpnet-base",
         "mxbai-embed-large-v1": "mxbai-large",
         "qwen3-embedding-0.6b": "qwen3-0.6b",
         "qwen3-embedding-4b": "qwen3-4b",
         "qwen3-embedding-8b": "qwen3-8b"}


def canon(k):
    """The lowercase short name for a checkpoint, whatever spelling it arrived under."""
    low = str(k).lower()
    return ALIAS.get(low, low)


nice = lambda k: NICE.get(canon(k), k)

# The evaluated set.  Training arms (arm_*), intervention checkpoints (ft_*) and other
# legacy keys accumulate in the e2 JSONs; they have their own tables and are not models
# under evaluation, so tables that report the evaluated set filter on this.
CORE = set(NICE.values())


# ---------------------------------------------------------------- number formatting
def f(v, n=3):
    return "--" if v is None else f"{v:.{n}f}"


def sf(v, n=3):
    return "--" if v is None else f"{v:+.{n}f}"


def pct(v, n=1):
    return "--" if v is None else f"{100 * v:.{n}f}%"


def sci(v):
    return "--" if v is None else f"{v:.1e}"


def ci(pair, n=3):
    if not pair or pair[0] is None:
        return "--"
    return f"{pair[0]:.{n}f}-{pair[1]:.{n}f}"


def Table(id, title, source, head, rows, note=""):
    return dict(id=id, title=title, source=source, head=head, rows=rows, note=note)


# =============================================================================
# One function per table. Each returns a Table, or None when its source is absent.
# =============================================================================

def t01_passive(D):
    r = D.get("ratios") or {}
    if not r:
        return None
    rows = [[nice(k), f(v["d_kept"], 4), f(v["d_reversed"], 4), f(v["ratio"]),
             ci(v["ci90"]), sci(v["p_not_further"])]
            for k, v in sorted(r.items(), key=lambda kv: kv[1]["ratio"] or 9)]
    return Table("T01", "The reversal is closer than the paraphrase", "e2 minimal-pairs",
                 ["model", "d to kept meaning", "d to reversed", "ratio", "90% CI",
                  "P not further"], rows,
                 "Passive contrast, verb held fixed. Distance = 1 - centred cosine. "
                 "A ratio below 1 means the sentence that reverses who acts on whom sits "
                 "closer than the one that preserves the meaning in other words.")


def t01b_precision(D):
    r = D.get("precision") or {}
    if not r:
        return None
    rows = []
    for key in sorted(r, key=lambda k: (k.split("@")[0], k.split("@")[1])):
        model, prec = key.split("@", 1)
        cell = r[key]
        rows.append([nice(model), prec,
                     f(cell.get("passive", {}).get("ratio")),
                     ci(cell.get("passive", {}).get("ci90")),
                     f(cell.get("twins", {}).get("ratio")),
                     ci(cell.get("twins", {}).get("ci90"))])
    return Table("T01b", "The scale trend is not an artefact of precision", "e2 minimal-pairs",
                 ["model", "precision", "passive ratio", "90% CI", "twin ratio", "90% CI"], rows,
                 "Sizes in a family ship at different default precisions and quantisation "
                 "drives the ratio toward 1, in the same direction as the trend with size, so "
                 "the two are crossed rather than left at each default. Compare down a column, "
                 "at a common precision.")


def t02_equivalence(D):
    r = D.get("equivalence") or {}
    if not r:
        return None
    # the matched-twin design logs cosines directly; fall back to 1 - d if it ever does not
    cos = lambda v, side: (v.get(f"cos_{side}") if v.get(f"cos_{side}") is not None
                           else (None if v.get(f"d_{side}") is None else 1 - v[f"d_{side}"]))
    rows = [[nice(k), f(cos(v, "kept"), 4), f(cos(v, "reversed"), 4), f(v["ratio"]),
             ci(v["ci90"]), sci(v["p_not_further"]), sci(v["p_tost"]),
             "yes" if v.get("role_blind") else "NO"]
            for k, v in sorted(r.items(), key=lambda kv: kv[1]["ratio"] or 9)]
    return Table("T02", "Equivalence tests on the verb contrast", "e2 minimal-pairs",
                 ["model", "kept cos", "reversed cos", "ratio", "90% CI", "P not further",
                  "P alike", "role-blind"], rows,
                 "Bounds 0.80-1.25, fixed before any model was run (the bioequivalence "
                 "convention). P alike tests the stricter claim that the two are treated "
                 "the same, not merely that the reversal is not further. Cosines, not "
                 "distances: the contrast is recorded as cosine in the protocol.")


def t02b_percell(D):
    r = D.get("percell") or {}
    if not r:
        return None
    rows = [[nice(k), f(v["ratio"]), ci(v["ci90"]), sci(v["p_not_further"]),
             sci(v["p_tost"]), "yes" if v.get("role_blind") else "NO"] for k, v in r.items()]
    return Table("T02b", "Per-cell evidence, predicate excluded", "e2 minimal-pairs",
                 ["model", "reversed / kept", "90% CI", "P not further", "P alike",
                  "role-blind"], rows,
                 "The same equivalence test applied to the explanation's own evidence "
                 "rather than to the similarity. It replaces the withdrawn half-strength "
                 "claim: the evidence for a reversed pair is not weaker, it is the same.")


def t03_probe(D):
    p = D.get("probe") or {}
    if not p:
        return None
    cols = ["act_act", "act_pas", "both_act", "both_pas", "ctrl_seen", "ctrl_new"]
    calib = [k for k in p if k.startswith("calibration")]
    rows = [[k if k.startswith("calibration") else nice(k)] + [pct(p[k][c]) for c in cols]
            for k in calib + [k for k in p if k not in calib]]
    return Table("T03", "A linear probe reads the roles off the representation", "e2 probe",
                 ["features", "act->act", "act->pas", "both->act", "both->pas",
                  "control seen", "control new"], rows,
                 "Tested on name pairs never seen in training. An order code alone cannot "
                 "exceed 50% on both voices, so both->act and both->pas above 50% together "
                 "mean names are bound to roles. The control task (Hewitt-Liang, random "
                 "labels) near 50% excludes memorisation.")


def t04_layers(D):
    L = D.get("layers") or {}
    if not L:
        return None
    rows = []
    for model, rs in L.items():
        for r in rs:
            rows.append([nice(model), r["layer"], pct(r["probe_active"]),
                         pct(r["probe_passive"]), f(r["ratio"]), ci(r["ci90"])])
    return Table("T04", "No layer's similarity separates a role reversal", "e2 layers",
                 ["model", "layer", "probe active", "probe passive", "distance ratio",
                  "90% CI"], rows,
                 "Layer 0 is a constant representation under CLS pooling, so no distance "
                 "exists there and it is omitted. A ratio above 1 at any layer would mean "
                 "reading that layer repairs the failure without retraining.")


def t05_measures(D):
    m = D.get("measures") or {}
    if not m:
        return None
    runs = {}
    for dataset, per_model in m.items():
        for model, r in per_model.items():
            if not isinstance(r, dict):
                continue
            name = nice(model)
            if name not in CORE:
                continue                      # arm_*, ft_*: reported in T07 and T10
            g = lambda k: r.get(k)
            pair = lambda t, n=3: ("--" if not isinstance(t, (list, tuple)) or t[0] is None
                                   else f"{t[0]:.{n}f}/{t[1]:.{n}f}")
            flat = lambda t, n=3: (f(t, n) if isinstance(t, (int, float)) else pair(t, n))
            cells = [r.get("n"), flat(g("cosine")), flat(g("crosspos")),
                     flat(g("percell"), 2), flat(g("overlap_mass")),
                     flat(g("mass_vs_score")), pct(g("faithfulness"), 0)]
            runs.setdefault((dataset, name), []).append((model, cells))

    # One model, two runs, two sets of numbers: keep both and label them by the key they
    # came under, rather than silently picking one.  A clash is a data problem, not a
    # formatting one, so say so at generation time.
    rows, clashes = [], []
    for dataset, name in sorted(runs):
        got = runs[(dataset, name)]
        if len({tuple(str(c) for c in cells) for _, cells in got} ) == 1:
            rows.append([dataset, name] + got[0][1])
        else:
            clashes.append((dataset, name, [k for k, _ in got]))
            for raw, cells in got:
                rows.append([dataset, f"{name} ({raw})"] + cells)
    if clashes:
        print("!" * 74)
        print("! T05: one model appears under more than one key, with DIFFERENT numbers.")
        for d, n, keys in clashes:
            print(f"!   {d:9s} {n:14s} <- {', '.join(keys)}")
        print("! Both runs are kept above and labelled by key. Decide which run is canonical,")
        print("! drop the other from the e2 measures JSON, or rerun e2 measures for the set.")
        print("!" * 74)
    return Table("T05", "Explanation measures, by dataset", "e2 measures",
                 ["dataset", "model", "n", "cosine", "cross-position share", "per-cell ratio",
                  "overlap mass", "mass vs score", "faithfulness"], rows,
                 "Where two values are separated by a slash they are class 1 / class 0 "
                 "(paraphrase or meaning kept / non-paraphrase or roles reversed). STS-B is "
                 "graded rather than classed, so its values are over all pairs and the "
                 "mass-vs-score column is how far identical-word evidence tracks the human "
                 "score. NOT the same quantity as the partial-Spearman table (F6), which is "
                 "still legacy output and needs the e1 rerun.")


def t06_graded(D):
    g = D.get("graded") or {}
    before, after = g.get("before") or {}, g.get("after") or {}
    if not after:
        return None
    rows = [["base", "--", f(before.get("stsb_rho")), sf(before.get("overlap_excess")),
             f(before.get("fooled")), "--", "--"]]
    for tag, r in after.items():
        mc = r.get("mcnemar") or {}
        cond = "shuffled" if "shuffled" in tag else "real scores"
        seed = tag.rsplit("_s", 1)[-1]
        rows.append([cond, seed, f(r.get("stsb_rho")), sf(r.get("overlap_excess")),
                     f(r.get("fooled")), f"{mc.get('fixed')} / {mc.get('broke')}",
                     sci(mc.get("p"))])
    return Table("T06", "Graded supervision and its shuffled control", "train graded",
                 ["condition", "seed", "STS-B", "overlap excess", "fooled",
                  "queries fixed / broken", "McNemar P"], rows,
                 "Identical data, steps and schedule; the control shuffles the scores. "
                 "Separated across seeds: the excess and STS-B. NOT separated: the fooled "
                 "rate. Do not state a causal link from removing the excess to fixing "
                 "retrieval.")


def t07_negatives(D):
    n = D.get("negatives") or {}
    before, after = n.get("before") or {}, n.get("after") or {}
    if not after:
        return None
    rows = [["base", "--", "--", f(before.get("stsb_rho")),
             sf(before.get("overlap_excess")), f(before.get("fooled"))]]
    for tag, r in after.items():
        rows.append([r.get("kind", tag), r.get("seed"), f(r.get("negative_overlap")),
                     f(r.get("stsb_rho")), sf(r.get("overlap_excess")), f(r.get("fooled"))])
    return Table("T07", "What the negative has to be", "train negatives",
                 ["negative", "seed", "its overlap with the anchor", "STS-B",
                  "overlap excess", "fooled"], rows,
                 "If word matching survives because nothing in training punishes it, a "
                 "negative that shares the anchor's words should be the one that removes it.")


def t08_checkpoints(D):
    e = D.get("evaluate") or {}
    if not e:
        return None
    rows = [[nice(k), f(v["top1_clean"]), f(v["top1_swap"]), f(v["fooled"]),
             f(v["stsb_rho"]), sf(v["overlap_excess"])] for k, v in e.items()]
    return Table("T08", "Retrieval and benchmark scores per checkpoint", "train evaluate",
                 ["checkpoint", "top-1 clean", "top-1 with swap", "fooled", "STS-B",
                  "overlap excess"], rows,
                 "A fooled rate that falls while top-1 clean also falls is a worse "
                 "retriever, not a repaired one. Read the two columns together.")


def t09_directions(D):
    d = (D.get("entangle") or {}).get("directions") or {}
    if not d:
        return None
    rows = [[nice(k), v.get("n_names"), f(v["shared_axis"]), f(v["random_level"]),
             f(v["own_name"]), f(v["other_name"]), pct(v["variance_top"], 0)]
            for k, v in d.items()]
    return Table("T09", "Roles are encoded per entity, not on a shared axis", "e4 entanglement",
                 ["model", "names", "role vs role", "random level", "own name",
                  "other names", "variance in top directions"], rows,
                 "Role direction of a name = that name as agent minus that name as patient. "
                 "Role directions align with each other barely above the random level, and "
                 "each rides on its own name's identity direction. An edit would need a "
                 "different direction for every entity.")


def t10_exchange(D):
    e = (D.get("entangle") or {}).get("exchange") or {}
    if not e:
        return None
    rows = []
    for model, rs in e.items():
        # by setting, then by edit, so the two kinds sit adjacent and can be compared
        for r in sorted(rs, key=lambda x: (-(x.get("rank") or 0), -(x.get("lam") or 0),
                                           str(x.get("edit")))):
            rows.append([nice(model), r.get("edit"), r.get("rank"), r.get("lam"),
                         f(r.get("removed"), 4), f(r.get("stsb_lost"), 4), f(r.get("rate"))])
    return Table("T10", "What separating role swaps costs", "e4 entanglement",
                 ["model", "edit", "rank", "lambda", "similarity removed", "STS-B lost",
                  "STS-B per 0.01 removed"], rows,
                 "Compare rows only at the same rank and lambda: each lambda removes a "
                 "different amount, so a rate averaged across them is not a like-for-like "
                 "comparison. Read with T09 and T14: the rank-64 edit lacked the capacity to span the "
                 "binding, and the rank-768 edit had it and failed on cost instead.")


def t11_cost(D):
    c = D.get("cost") or {}
    if not c:
        return None
    ds = list(next(iter(c.values())).get("datasets", {}))
    order = ["retrained (baseline)", "none", "random", "topic-matched", "word-shuffled",
             "non-role swap", "role swap", "graded, real", "graded, shuffled"]
    keys = [k for k in order if k in c] + [k for k in c if k not in order]
    rows = []
    for k in keys:
        r = c[k]
        ch = r.get("change")
        rows.append([k] + [f(r["datasets"][d]) for d in ds] + [f(r.get("mean")),
                    "--" if ch is None else f"{100 * ch:+.1f}%", f(r.get("fooled"))])
    return Table("T11", "What the correction costs zero-shot retrieval", "e3 nanobeir",
                 ["condition"] + ds + ["mean", "change", "fooled"], rows,
                 "NanoBEIR nDCG@10, all conditions trained identically. One seed each "
                 "except where the caption gives three. The benefit is stable across seeds "
                 "while the cost varies by a factor of two and a half, so the damage looks "
                 "incidental rather than intrinsic.")


def t12_superposition(D):
    p = (D.get("superpos") or {}).get("per_model") or {}
    if not p:
        return None
    rows = [[nice(k), f(v["binding_fraction"]), f(v["variance_share"]), f(v["across_pairs"]),
             f(v["random_level"]), f(v["own_entity"]), f(v["other_entities"])]
            for k, v in p.items()]
    return Table("T12", "The binding term, decomposed", "e5 decompose",
                 ["model", "|D|/|S|", "variance share", "across pairs", "random level",
                  "own entity", "other entities"], rows,
                 "Fixed frame, verb, year and city throughout, so the entity pair is the "
                 "only thing that varies. Quote absolute cosines from the varied-frame run "
                 "instead: centring differs between the two and they are not comparable.")


def t13_scaling(D):
    sc = (D.get("superpos") or {}).get("scaling") or {}
    if not sc:
        return None
    sizes = [r["n_names"] for r in next(iter(sc.values()))]
    rows = [["a generic role axis would need"] + ["1"] * len(sizes)]
    rows += [[nice(k)] + [str(r["rank_90"]) for r in rs] for k, rs in sc.items()]
    ceiling = next(iter(sc.values()))[0].get("max_possible")
    return Table("T13", "The rank of the binding term grows with the vocabulary", "e5 decompose",
                 ["model"] + [f"{n} entities" for n in sizes], rows,
                 f"Directions holding 90% of the binding term, at a fixed item count "
                 f"(ceiling {ceiling} in every column, so the rank is nowhere near the "
                 f"sample bound). Growth is sub-linear and will saturate; we cannot say "
                 f"where. Do not extrapolate to the embedding dimension.")


def _blind_head(B, h):
    v = B.get(h) or {}
    return f"{h}: {'HELD' if v.get('holds') else 'DOES NOT HOLD'} -- registered: " \
           f"{v.get('prediction', '')}"


def t14_h1(D):
    B = D.get("blind") or {}
    per = (B.get("H1") or {}).get("per_model") or {}
    if not per:
        return None
    subs = sorted({c for v in per.values() for c in (v.get("per_subcase") or {})})
    rows = [[nice(k), f(v["overall"]), ci(v.get("ci"))] +
            [f((v.get("per_subcase") or {}).get(c)) for c in subs]
            for k, v in per.items()]
    return Table("T14", "H1  HANS entailment", "e6 blind",
                 ["model", "overall AUC", "90% CI"] + subs, rows,
                 _blind_head(B, "H1") + ". AUC is the probability a true entailment scores "
                 "above a non-entailment. We predicted indifference; below 0.5 the "
                 "similarity prefers the role-swapped non-entailment. Report the miss as a "
                 "miss: the effect exceeded the prediction.")


def t15_h2(D):
    B = D.get("blind") or {}
    per = (B.get("H2") or {}).get("per_model") or {}
    if not per:
        return None
    langs = list(next(iter(per.values())).get("per_language", {}))
    rows = [[nice(k)] + [f(v["per_language"][l]) for l in langs] +
            [sf(v.get("max_above_english"))] for k, v in per.items()]
    return Table("T15", "H2  PAWS-X, the deficit is not an English artefact", "e6 blind",
                 ["model"] + langs + ["max - English"], rows,
                 _blind_head(B, "H2") + ". Every language sits below English, so nothing "
                 "approaches the bound. English AUC also rises with size, a second instance "
                 "of the scale trend on data never used in development.")


def t16_h3(D):
    B = D.get("blind") or {}
    H = B.get("H3") or {}
    per = H.get("per_model") or {}
    if not per:
        return None
    sets = list(next(iter(per.values())).get("excess", {}))
    rows = []
    if H.get("cross_encoder"):
        rows.append(["cross-encoder (reference)"] + [f(H["cross_encoder"][s]) for s in sets] + ["--"])
    rows += [[nice(k)] + [sf(v["excess"][s]) for s in sets] + [f"{v['n_positive']}/{len(sets)}"]
             for k, v in per.items()]
    return Table("T16", "H3  the benchmark excess generalises", "e6 blind",
                 ["model"] + sets + ["positive"], rows,
                 _blind_head(B, "H3") + ". The excess here is larger than on STS-B, so the "
                 "benchmark-reliance claim generalises and grows. The graded model was tuned "
                 "on STS-B train, so these six sets are out of domain for it.")


def t17_h4(D):
    B = D.get("blind") or {}
    H = B.get("H4") or {}
    series = [(k, v) for k, v in H.items() if isinstance(v, list)]
    if not series:
        return None
    n = max(len(v) for _, v in series)
    rows = [[k] + [f(v[i]) if i < len(v) else "--" for i in range(n)] for k, v in series]
    return Table("T17", "H4  does the specificity transfer?", "e6 blind",
                 ["condition"] + [f"seed {i + 1}" for i in range(n)], rows,
                 _blind_head(B, "H4") + ". Entity-swap training transfers strongly, but the "
                 "edit-matched control transfers too and the ranges overlap. A real "
                 "limitation: on HANS, any negative that teaches word order to matter helps, "
                 "because that is what its subject/object subcase tests. Scope the claim to "
                 "the measure.")


def t18_examples(D):
    ex = D.get("examples") or []
    if not ex or all(e.get("demo_only") for e in ex):
        return None          # synthetic layout data is not a table
    models = sorted({m for e in ex for m in e["per_model"]})
    rows = []
    for e in ex:
        for m in models:
            r = e["per_model"].get(m)
            if not r:
                continue
            rows.append([e["domain"], nice(m), e["original"], e["kept"], e["reversed"],
                         f(r["d_kept"], 4), f(r["d_reversed"], 4), f(r["ratio"]),
                         r["closer"]])
    return Table("T18", "Every worked example, every model", "fig_examples",
                 ["domain", "model", "original", "same meaning / new words",
                  "opposite meaning / same words", "d kept", "d reversed", "ratio",
                  "closer"], rows,
                 "All items are reported, including those where the paraphrase correctly "
                 "wins. The items were written before they were run and none was replaced "
                 "after seeing a result.")


TABLES = [t01_passive, t01b_precision, t02_equivalence, t02b_percell, t03_probe, t04_layers, t05_measures, t06_graded,
          t07_negatives, t08_checkpoints, t09_directions, t10_exchange, t11_cost,
          t12_superposition, t13_scaling, t14_h1, t15_h2, t16_h3, t17_h4, t18_examples]


# =============================================================================
# Writing
# =============================================================================

def slug(title):
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:44]


def to_markdown(t):
    out = [f"## {t['id']}. {t['title']}", f"*Source: {t['source']}*", ""]
    out.append("| " + " | ".join(str(h) for h in t["head"]) + " |")
    out.append("|" + "|".join("---" for _ in t["head"]) + "|")
    for r in t["rows"]:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    if t["note"]:
        out += ["", t["note"]]
    return "\n".join(out) + "\n"


def tex_escape(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s.replace("->", r"$\rightarrow$")


def to_latex_body(t):
    """Just the tabular, so a manuscript can supply its own caption and placement.
    The generated caption is kept in the standalone file; this one is for \\input."""
    cols = "l" + "r" * (len(t["head"]) - 1)
    out = [r"% " + f"{t['id']}  {t['title']}  (generated, do not edit)",
           r"\begin{tabular}{" + cols + "}", r"\toprule",
           " & ".join(tex_escape(h) for h in t["head"]) + r" \\", r"\midrule"]
    out += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in t["rows"]]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def to_latex(t):
    cols = "l" + "r" * (len(t["head"]) - 1)
    out = [r"% " + f"{t['id']}  {t['title']}  (source: {t['source']})",
           r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{" + tex_escape(t["title"]) + ". " + tex_escape(t["note"]) + "}",
           r"\begin{tabular}{" + cols + "}", r"\toprule",
           " & ".join(tex_escape(h) for h in t["head"]) + r" \\", r"\midrule"]
    out += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in t["rows"]]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def stale_warning(_=None):
    """figdata.json is a snapshot. When the collector has changed since it was written,
    a table can be missing for no reason other than that figdata.py has not been rerun.
    That has caught us three times, so say it out loud."""
    collector = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figdata.py")
    if not (os.path.exists(IN) and os.path.exists(collector)):
        return
    if os.path.getmtime(collector) > os.path.getmtime(IN):
        print("!" * 74)
        print("! figdata.py is NEWER than results/figs/figdata.json.")
        print("! Anything the collector learned to read since then is not in that file,")
        print("! so a table may be reported as missing when its data is on disk.")
        print("!")
        print("!     python figdata.py && python tables.py --check")
        print("!" * 74 + "\n")


# ----------------------------------------------------------------- one review PDF
PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,landscape,margin=12mm]{geometry}
\usepackage{booktabs,longtable,array,xcolor}
\usepackage[colorlinks,linkcolor=blue]{hyperref}
\renewcommand{\arraystretch}{1.12}
\setlength{\parskip}{0.4em}\setlength{\parindent}{0pt}
\title{\vspace{-14mm}All tables}
\date{}
\begin{document}
\maketitle
%s
\tableofcontents
\clearpage
"""


def colspec(t, long_at=28, width="4.6cm"):
    """Right-align numbers; wrap any column whose cells are long, so that a table of
    sentences stays readable instead of running off the page."""
    out = []
    for j, head in enumerate(t["head"]):
        cells = [str(r[j]) for r in t["rows"]] + [str(head)]
        longest = max(map(len, cells))
        if longest > long_at:
            # ragged right: justifying a narrow column of sentences leaves rivers of
            # white space and a pageful of underfull-box warnings
            out.append(r">{\raggedright\arraybackslash}p{%s}" % width)
        elif j == 0:
            out.append("l")
        else:
            out.append("r")
    return "".join(out)


def to_longtable(t):
    """One table as a longtable, so it may break across pages and keep its header."""
    nl = r" \\"
    head = " & ".join(tex_escape(h) for h in t["head"]) + nl
    title = f"{t['id']}. " + tex_escape(t["title"])
    out = [r"\section*{" + title + "}",
           r"\addcontentsline{toc}{section}{" + title + "}",
           r"\textcolor{gray}{Source: " + tex_escape(t["source"])
           + r". Generated by tables.py; do not edit by hand.}", "",
           r"\footnotesize",
           r"\begin{longtable}{" + colspec(t) + "}",
           r"\toprule", head, r"\midrule", r"\endfirsthead",
           r"\toprule", head, r"\midrule", r"\endhead",
           r"\bottomrule", r"\endfoot"]
    out += [" & ".join(tex_escape(c) for c in r) + nl for r in t["rows"]]
    out += [r"\end{longtable}", r"\normalsize", ""]
    if t["note"]:
        out.append(r"\textit{" + tex_escape(t["note"]) + "}")
    out.append(r"\clearpage")
    return "\n".join(out) + "\n"


def write_review(built, path, demo):
    banner = (r"\begin{center}\textcolor{red}{\textbf{DEMO DATA --- not a collection}}"
              r"\end{center}") if demo else ""
    with open(path, "w") as fh:
        fh.write(PREAMBLE % banner)
        fh.write("\n".join(to_longtable(t) for t in built))
        fh.write("\n" + r"\end{document}" + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="table ids, e.g. T01 T11")
    ap.add_argument("--check", nargs="?", const="TABLES.md",
                    help="a reference file to check every number against "
                         "(default TABLES.md)")
    args = ap.parse_args()

    stale_warning(args.only)
    src = next((p for p in (IN, IN_DEMO) if os.path.exists(p)), None)
    if src is None:
        raise SystemExit(f"{IN} not found -- run `python figdata.py` first")
    with open(src) as fh:
        D = json.load(fh)
    demo = D.get("meta", {}).get("source") == "demo"
    if not demo and not any(D.get(k) for k in ("ratios", "probe", "blind")) \
            and os.path.exists(IN_DEMO):
        with open(IN_DEMO) as fh:     # a real collection that found nothing yet
            D = json.load(fh)
        demo = D.get("meta", {}).get("source") == "demo"
    print(f"reading {src if not demo else IN_DEMO}\n")
    if demo:
        print("!! DEMO DATA: these tables are the logged numbers re-emitted, not a fresh "
              "collection. Run `python figdata.py` before using them.\n")

    os.makedirs(OUT, exist_ok=True)
    built, missing = [], []
    for fn in TABLES:
        t = fn(D)
        if t is None:
            missing.append(fn.__name__)
            continue
        if args.only and t["id"] not in args.only:
            continue
        built.append(t)
        stem = os.path.join(OUT, f"{t['id']}_{slug(t['title'])}")
        with open(stem + ".csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(t["head"])
            w.writerows(t["rows"])
        with open(stem + ".tex", "w") as fh:
            fh.write(to_latex(t))
        # body only, for \input into a manuscript that writes its own caption
        with open(os.path.join(OUT, f"{t['id']}_body.tex"), "w") as fh:
            fh.write(to_latex_body(t))
        print(f"  {t['id']}  {len(t['rows']):4d} rows   {t['title']}")

    banner = ("<!-- DEMO DATA, not a collection -->\n\n" if demo else "")
    with open(os.path.join(OUT, "ALL_TABLES.md"), "w") as fh:
        fh.write(banner + "# Tables\n\nBuilt by tables.py from "
                 f"{D.get('meta', {}).get('written', '?')}. Every number comes from a run's "
                 "own JSON; none is typed in.\n\n")
        fh.write("\n\n".join(to_markdown(t) for t in built))

    write_review(built, os.path.join(OUT, "ALL_TABLES.tex"), demo)
    print(f"\n{len(built)} tables written to {OUT}/  (csv, tex, _body.tex, "
          f"ALL_TABLES.md and ALL_TABLES.tex)")
    print(f"One PDF of every table:\n  cd {OUT} && pdflatex ALL_TABLES.tex && "
          f"pdflatex ALL_TABLES.tex")
    if missing:
        ids = ["T" + m.split("_")[0][1:] for m in missing]     # t02b_percell -> T02b
        print("not built, source not collected: " + ", ".join(ids))
        print("Each of those is a run that has not happened. None is filled in.")

    if args.check:
        if not os.path.exists(args.check):
            print(f"\n[check] {args.check} not found -- skipped")
        else:
            check(built, args.check)


# ----------------------------------------------------------------- checking
# A cell is checkable when it is a decimal or a percentage. Integers are not: a bare
# "1" or "42" matches everywhere and would tell us nothing. Scientific notation, ranges
# and text are counted as skipped rather than passed, so the coverage is visible.
DECIMAL = re.compile(r"[+-]?\d+\.\d{2,4}$")
PERCENT = re.compile(r"[+-]?\d+\.\d+%$")


def normalise(text):
    """En and em dashes read as hyphens, and a minus sign as a hyphen, so a range or a
    negative written for print still matches what the code prints."""
    for a, b in (("\u2013", "-"), ("\u2014", "-"), ("\u2212", "-")):
        text = text.replace(a, b)
    return text


def check(built, path):
    with open(path) as fh:
        ref = normalise(fh.read())
    missing, checked, skipped = [], 0, 0
    for t in built:
        for r in t["rows"]:
            for c in r:
                v = normalise(str(c)).strip()
                if not (DECIMAL.match(v) or PERCENT.match(v)):
                    skipped += 1
                    continue
                checked += 1
                # a positive value may be written with or without its sign
                forms = {v, v.lstrip("+")} if v.startswith("+") else {v, "+" + v}
                if not any(x in ref for x in forms):
                    missing.append((t["id"], v))

    print(f"\n[check] against {path}")
    print(f"  {checked} values checked, {skipped} not checkable "
          f"(integers, ranges, scientific notation, text)")
    print(f"  {len(missing)} not found")
    if not missing:
        print("  Every checkable number in the logs appears in the reference.")
        return
    by_table = {}
    for tid, v in missing:
        by_table.setdefault(tid, []).append(v)
    for tid in sorted(by_table):
        vs = by_table[tid]
        print(f"  {tid}  {len(vs):3d}  " + ", ".join(vs[:8]) + (" ..." if len(vs) > 8 else ""))
    print("\nA value in the logs but not in the reference is one of three things: a number "
          "that\nmoved under the clean core, a transcription error in the reference, or a "
          "cell the\nreference never recorded. Resolve each before writing; none of them is "
          "cosmetic.")


if __name__ == "__main__":
    main()

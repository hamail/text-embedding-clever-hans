#!/usr/bin/env python
"""figdata.py -- gather what the experiments already logged into one figure-ready file.

No model is loaded and nothing is measured again. Every number a figure draws comes from
the JSON an experiment wrote, so a figure cannot quietly disagree with the logs. Run this,
read the coverage report, and rerun an experiment only where a line says MISSING.

  python figdata.py                collect from results/  -> results/figs/figdata.json
  python figdata.py --demo         the logged tables instead, so layout can be checked
                                   without a GPU -> results/figs/figdata_demo.json, and
                                   every figure drawn from it is stamped DEMO
  python figdata.py --strict       exit non-zero if any required source is missing

The two never overwrite each other. The notebook loads figdata.json when it exists and
falls back to the demo, so the real numbers always win once they are collected.

Writes this shape:

  meta      source ("runs" or "demo"), written, and the files each block came from
  examples  [{domain, original, kept, reversed, per_model: {m: {d_kept, d_reversed,
            ratio, closer}}}]                                   <- fig_examples.py
  ratios    {model: {d_kept, d_reversed, ratio, ci90, p_not_further, p_tost,
            role_blind}}                                        <- e2 minimal-pairs
  probe     {feature set: {act_act, act_pas, both_act, both_pas, ctrl_seen, ctrl_new}}
                                                                <- e2 probe
  layers    {model: [{layer, probe_active, probe_passive, ratio, ci90}]}
                                                                <- e2 layers
  entangle  {model: {shared_axis, random_level, own_name, other_name, variance_top}}
            exchange {model: [{edit, rank, lam, removed, stsb_lost, rate}]}
                                                                <- e4 entanglement
  superpos  {model: {binding_fraction, variance_share, across_pairs, random_level,
            own_entity, other_entities, rank_90}}
            scaling {model: [{n_names, rank_90, max_possible, n_items}]}
                                                                <- e5 decompose
  blind     {H1..H4: as e6 wrote them}                          <- e6
  cost      {condition: {datasets: {...}, mean, change, fooled}} <- e3 nanobeir
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

OUT = "results/figs"

# Where each block comes from, and whether a figure depends on it.
#   key          glob                              required by
SOURCES = {
    "examples": ("results/figs/examples.json", "figs 1, 2"),
    "ratios": ("results/e2/minimal_pairs*.json", "fig 2"),
    "probe": ("results/e2/probe*.json", "fig 3"),
    "layers": ("results/e2/layers.json", "fig 3"),
    "entangle": ("results/e4/entanglement.json", "fig 4"),
    "superpos": ("results/e5/superposition.json", "fig 4"),
    "blind": ("results/blind/blind_results.json", "fig 5"),
    "cost": ("results/e3/nanobeir.json", "table 1"),
    "graded": ("results/train/graded.json", "tables F7"),
    "negatives": ("results/train/negatives.json", "tables F9, F10"),
    "evaluate": ("results/train/evaluate.json", "tables F8 context"),
}

# Keyed by the dataset in the filename rather than merged, since measures_stsb and
# measures_mrpc are different tables, not different models of one table.
MEASURES = "results/e2/measures_*.json"

PROBE_COLS = {
    "act_act": "trained on active -> active",
    "act_pas": "trained on active -> passive",
    "both_act": "trained on both voices -> active",
    "both_pas": "trained on both voices -> passive",
    "ctrl_seen": "control task -> familiar pairs",
    "ctrl_new": "control task -> unseen pairs",
}


# =============================================================================
# Reading
# =============================================================================

def merged(pattern: str, exclude: str = "") -> tuple[dict, list[str]]:
    """Every file matching the pattern, merged. A run with a --tag writes its own file,
    so several files can hold different models for one block; later files win a clash.
    Files that wrap their payload in "results" are unwrapped, others passed through.

    exclude drops files whose name contains it.  The crossed-precision runs write
    minimal_pairs_p_<size>_<prec>.json, which the ratios glob also matches, and because
    later files win, qwen3-0.6b's headline ratio silently became whichever precision
    sorted last (int8, 0.465) instead of its default (0.442).  Those cells belong to
    collect_precision and T01b, so they are kept out of the model tables."""
    out: dict = {}
    used: list[str] = []
    for path in sorted(glob.glob(pattern)):
        if exclude and exclude in os.path.basename(path):
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"    ! unreadable, skipped: {path} ({exc})")
            continue
        used.append(path)
        payload = data.get("results", data) if isinstance(data, dict) else data
        if isinstance(payload, dict):
            out.update(payload)
        else:
            out.setdefault("_list", []).extend(payload)
    return out, used


# =============================================================================
# Normalising: each takes the raw merge and returns the documented shape.
# Every field is fetched with .get so a partial run degrades instead of crashing.
# =============================================================================

def norm_examples(raw: dict) -> list[dict]:
    """fig_examples.py writes {items: [...], results: {model: [rows]}}; the merge above
    keeps the results, so the file is reread for the items alongside them."""
    path = SOURCES["examples"][0]
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        data = json.load(fh)
    items, per_model = data.get("items", []), data.get("results", {})
    out = []
    for k, it in enumerate(items):
        row = {f: it.get(f) for f in ("domain", "original", "kept", "reversed")}
        row["per_model"] = {}
        for model, rows in per_model.items():
            if k < len(rows):
                r = rows[k]
                row["per_model"][model] = {f: r.get(f) for f in
                                           ("d_kept", "d_reversed", "ratio", "closer")}
        out.append(row)
    return out


# What e2 minimal-pairs actually calls its two designs. "similarity" is the matched-twin
# design (the verb contrast) and carries cosines; "passive contrast" holds the verb fixed
# and carries distances. Reading a key that does not exist is silent, so the loader below
# prints the keys it did find.
PASSIVE, EQUIV, PERCELL = "passive contrast", "similarity", "per-cell evidence"


def norm_ratios(raw: dict, contrast: str = PASSIVE) -> dict:
    """One contrast out of the nest. Either design may carry distances or cosines, so both
    pairs of fields are kept and the tables take whichever is present."""
    keep = ("d_kept", "d_reversed", "cos_kept", "cos_reversed", "ratio", "ci90",
            "p_not_further", "p_tost", "role_blind", "precision")
    out, seen = {}, set()
    for model, per_contrast in raw.items():
        if not isinstance(per_contrast, dict):
            continue
        seen.update(per_contrast)
        r = per_contrast.get(contrast)
        if isinstance(r, dict):
            out[model] = {f: r.get(f) for f in keep}
    if not out and seen:
        print(f"    ! no \"{contrast}\" in these files. They hold: "
              + ", ".join(f'"{k}"' for k in sorted(seen)))
    return out


def norm_percell(raw: dict) -> dict:
    """The per-cell evidence, written only for models whose matrices were explained."""
    keep = ("ratio", "ci90", "p_not_further", "p_tost", "role_blind")
    return {m: {f: per[PERCELL].get(f) for f in keep}
            for m, per in raw.items()
            if isinstance(per, dict) and isinstance(per.get(PERCELL), dict)}


def norm_probe(raw: dict) -> dict:
    out = {}
    for features, cols in raw.items():
        if not isinstance(cols, dict):
            continue
        row = {}
        for short, long in PROBE_COLS.items():
            cell = cols.get(long)
            row[short] = cell.get("accuracy") if isinstance(cell, dict) else None
        out[features] = row
    return out


def norm_layers(raw: dict) -> dict:
    out = {}
    for model, rows in raw.items():
        if not isinstance(rows, list):
            continue
        out[model] = [{f: r.get(f) for f in
                       ("layer", "probe_active", "probe_passive", "ratio", "ci90")}
                      for r in rows if not r.get("degenerate")]
    return out


def norm_entangle(raw: dict) -> dict:
    keep = ("n_names", "shared_axis", "random_level", "own_name", "other_name", "variance_top")
    dirs = {m: {f: v.get(f) for f in keep}
            for m, v in (raw.get("directions") or {}).items() if isinstance(v, dict)}
    exch = {}
    for model, rows in (raw.get("exchange_rates") or {}).items():
        if not isinstance(rows, list):
            continue
        exch[model] = [dict(edit=r.get("edit"), rank=r.get("rank"), lam=r.get("lam"),
                            removed=r.get("removed"), stsb_lost=r.get("stsb_lost"),
                            rate=(r["stsb_lost"] / (r["removed"] / 0.01)
                                  if r.get("removed") else None))
                       for r in rows]
    return dict(directions=dirs, exchange=exch)


def norm_superpos(raw: dict) -> dict:
    keep = ("binding_fraction", "variance_share", "cos_with_reversal", "across_pairs",
            "random_level", "own_entity", "other_entities", "rank_90", "n_pairs")
    per, scaling = {}, {}
    for model, r in raw.items():
        if not isinstance(r, dict):
            continue
        per[model] = {f: r.get(f) for f in keep}
        rows = r.get("scaling")
        if isinstance(rows, list):
            scaling[model] = [{f: s.get(f) for f in
                               ("n_names", "rank_90", "max_possible", "n_items")} for s in rows]
    return dict(per_model=per, scaling=scaling)


def norm_blind(raw: dict) -> dict:
    out = {h: raw[h] for h in ("H1", "H2", "H3", "H4") if h in raw}
    for tag in ("protocol_commit", "code_commit"):
        if tag in raw:
            out[tag] = raw[tag]
    return out


def norm_train(raw: dict) -> dict:
    """train.py writes {args, before, after:{tag: row}}; the merge unwraps nothing here,
    so before and after are taken as written."""
    return dict(before=raw.get("before", {}), after=raw.get("after", {}))


def norm_evaluate(raw: dict) -> dict:
    keep = ("top1_clean", "top1_swap", "fooled", "stsb_rho", "overlap_excess")
    return {k: {f: v.get(f) for f in keep} for k, v in raw.items() if isinstance(v, dict)}


def collect_precision() -> tuple[dict, list[str]]:
    """The precision-crossed runs write one file per cell and every file uses the same
    model key, so merging them keeps only the last and the comparison is lost. Key by
    model and precision instead. Precision is recorded on the matched-twin entry."""
    out, used = {}, []
    for path in sorted(glob.glob("results/e2/minimal_pairs_p_*.json")):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        used.append(path)
        for model, per in (data.get("results") or {}).items():
            if not isinstance(per, dict):
                continue
            prec = ((per.get(EQUIV) or {}).get("precision")
                    or (per.get(PASSIVE) or {}).get("precision") or "?")
            cell = {}
            for tag, key in (("passive", PASSIVE), ("twins", EQUIV)):
                r = per.get(key)
                if isinstance(r, dict):
                    cell[tag] = {f: r.get(f) for f in ("ratio", "ci90", "p_not_further")}
            if cell:
                out[f"{model}@{prec}"] = cell
    return out, used


def collect_measures() -> tuple[dict, list[str]]:
    """One entry per dataset, keyed by the suffix of the filename."""
    out, used = {}, []
    for path in sorted(glob.glob(MEASURES)):
        name = os.path.basename(path)[len("measures_"):-len(".json")]
        try:
            with open(path) as fh:
                out[name] = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"    ! unreadable, skipped: {path} ({exc})")
            continue
        used.append(path)
    return out, used


def norm_cost(raw: dict) -> dict:
    """e3 writes {condition: {dataset: ndcg, ...}} plus summary fields; kept as written,
    with the mean and change recomputed only where they are absent."""
    out = {}
    for cond, r in raw.items():
        if not isinstance(r, dict):
            continue
        out[cond] = r
    return out


NORMALISE = dict(examples=norm_examples, ratios=norm_ratios, probe=norm_probe,
                 layers=norm_layers, entangle=norm_entangle, superpos=norm_superpos,
                 blind=norm_blind, cost=norm_cost, graded=norm_train,
                 negatives=norm_train, evaluate=norm_evaluate)


# =============================================================================
# The demo stub: the logged tables, for checking layout only
# =============================================================================

def demo() -> dict:
    """Numbers copied from WRITING_NOTES, which were themselves copied from run output.
    This exists so a figure's layout can be checked on a machine with no GPU. It is
    marked so every figure drawn from it carries a DEMO stamp."""
    ex_models = ["unsup-simcse", "bge-base", "e5-base", "qwen3-0.6b", "qwen3-8b"]
    triples = [
        ("pharmacology", "Ritonavir inhibits the metabolism of midazolam.",
         "The metabolism of midazolam is slowed by ritonavir.",
         "Midazolam inhibits the metabolism of ritonavir.", 0.0492, 0.0098),
        ("molecular biology", "The kinase phosphorylates the receptor.",
         "The receptor is phosphorylated by the kinase.",
         "The receptor phosphorylates the kinase.", 0.0431, 0.0087),
        ("law", "The plaintiff sued the defendant for breach of contract.",
         "The defendant was sued by the plaintiff for breach of contract.",
         "The defendant sued the plaintiff for breach of contract.", 0.0388, 0.0071),
        ("clinical", "The donor's kidney was transplanted into the recipient.",
         "The recipient received a kidney from the donor.",
         "The recipient's kidney was transplanted into the donor.", 0.0605, 0.0113),
        ("epidemiology", "The virus was transmitted from the patient to the nurse.",
         "The nurse caught the virus from the patient.",
         "The virus was transmitted from the nurse to the patient.", 0.0702, 0.0126),
        ("negation", "The trial showed the drug reduces mortality.",
         "Mortality fell in the group given the drug.",
         "The trial showed the drug does not reduce mortality.", 0.0514, 0.0669),
    ]
    examples = []
    for dom, orig, kept, rev, dk, dr in triples:
        per = {}
        for i, m in enumerate(ex_models):
            s = 1.0 + 0.18 * i                      # a spread, so the panel is not flat
            a, b = dk * s, dr * s * (1.0 + 0.25 * i)
            per[m] = dict(d_kept=a, d_reversed=b, ratio=b / a,
                          closer="reversal" if b < a else "paraphrase")
        # No worked example has been measured yet, so these distances are synthetic: they
        # exist only to check that Figs 1 and 2 lay out, and they are marked so the table
        # of examples is not built from them.
        examples.append(dict(domain=dom, original=orig, kept=kept, reversed=rev,
                             per_model=per, demo_only=True))

    # F1 / F14, passive contrast
    r = [("retrained", 0.0277, 0.0045, 0.158, 0.153, 0.163),
         ("unsup-simcse", 0.0492, 0.0098, 0.184, 0.177, 0.191),
         ("e5-base", 0.0275, 0.0078, 0.265, 0.254, 0.276),
         ("sup-simcse", 0.0241, 0.0078, 0.312, 0.303, 0.321),
         ("bge-base", 0.0270, 0.0103, 0.346, 0.331, 0.360),
         ("mpnet-base", None, None, 0.380, 0.367, 0.393),
         ("mxbai-large", None, None, 0.419, 0.404, 0.434),
         ("gte-base", None, None, 0.426, 0.406, 0.446),
         ("qwen3-0.6b", 0.0834, 0.0506, 0.442, 0.423, 0.462),
         ("qwen3-4b", None, None, 0.602, 0.583, 0.623),
         ("qwen3-8b", None, None, 0.889, 0.858, 0.922)]
    ratios = {m: dict(d_kept=dk, d_reversed=dr, ratio=v, ci90=[lo, hi],
                      p_not_further=1e-200, p_tost=1e-10, role_blind=bool(hi < 1.25))
              for m, dk, dr, v, lo, hi in r}

    # F3
    probe = {}
    for k, v in {
        "calibration: bag of names": (.500, .500, .500, .500, 1.0, .551),
        "calibration: names by order": (1.0, .0, .500, .500, 1.0, .532),
        "calibration: names by role": (1.0, 1.0, 1.0, 1.0, 1.0, .532),
        "unsup-simcse": (.760, .261, .650, .680, .933, .523),
        "sup-simcse": (.759, .293, .668, .673, .924, .537),
        "bge-base": (.839, .269, .751, .729, .958, .507),
        "e5-base": (.816, .232, .721, .688, .957, .521),
        "retrained": (.680, .340, .608, .608, .952, .531),
        "qwen3-0.6b": (.982, .011, .806, .756, .958, .531),
    }.items():
        probe[k] = dict(zip(PROBE_COLS, v))

    # F4
    enc = [(5, .747, .706, .099), (6, .752, .707, .087),
           (7, .744, .708, .095), (8, .717, .688, .059), (9, .744, .634, .059),
           (10, .647, .649, .066), (11, .623, .658, .143), (12, .634, .643, .184)]
    # layers 1-4 of the encoder were never recorded in the notes, so they are absent
    # here rather than guessed; the real collection has every layer.
    dec = [(1, .556, .717, .009), (2, .732, .843, .066), (3, .761, .821, .085),
           (4, .746, .795, .103), (5, .778, .794, .097), (6, .763, .801, .098),
           (7, .762, .794, .102), (8, .762, .776, .149), (9, .754, .745, .139),
           (10, .744, .762, .157), (11, .753, .752, .154), (12, .747, .747, .161),
           (13, .743, .770, .168), (14, .794, .796, .167), (15, .790, .819, .166),
           (16, .770, .811, .175), (17, .804, .828, .177), (18, .804, .835, .179),
           (19, .786, .827, .183), (20, .798, .810, .190), (21, .799, .778, .197),
           (22, .803, .684, .295), (23, .781, .714, .302), (24, .772, .737, .303),
           (25, .778, .700, .352), (26, .751, .671, .419), (27, .768, .688, .443),
           (28, .768, .673, .442)]
    layers = {n: [dict(layer=L, probe_active=a, probe_passive=p, ratio=v,
                       ci90=[v * 0.96, v * 1.04]) for L, a, p, v in rows]
              for n, rows in (("unsup-simcse", enc), ("qwen3-0.6b", dec))}

    # F12
    entangle = dict(
        directions={m: dict(n_names=40, shared_axis=s, random_level=rl, own_name=own,
                            other_name=oth, variance_top=vt)
                    for m, s, rl, own, oth, vt in
                    [("unsup-simcse", .090, .030, .681, .087, .53),
                     ("retrained", .096, .030, .527, .089, .54),
                     ("qwen3-0.6b", .091, .025, .541, .056, .55)]},
        exchange={"unsup-simcse": [
            dict(edit="entity swaps", rank=768, lam=10, removed=.0508, stsb_lost=.1322, rate=.026),
            dict(edit="entity swaps", rank=64, lam=10, removed=.0364, stsb_lost=.0985, rate=.027),
            dict(edit="control swaps", rank=768, lam=10, removed=.1398, stsb_lost=.2118, rate=.015),
            dict(edit="control swaps", rank=64, lam=10, removed=.0988, stsb_lost=.1688, rate=.017)],
            "retrained": [
            dict(edit="control swaps", rank=768, lam=10, removed=.0926, stsb_lost=.1611, rate=.017),
            dict(edit="control swaps", rank=64, lam=10, removed=.0652, stsb_lost=.1191, rate=.018)]})

    # F16b / F16c
    sp = {m: dict(binding_fraction=bf, variance_share=vs, cos_with_reversal=None,
                  across_pairs=ap, random_level=rl, own_entity=oe, other_entities=ot,
                  rank_90=None, n_pairs=7140)
          for m, bf, vs, ap, rl, oe, ot in
          [("unsup-simcse", .077, .006, .118, .029, .249, .081),
           ("sup-simcse", .154, .023, .123, .029, .612, .081),
           ("bge-base", .123, .015, .117, .029, .554, .060),
           ("e5-base", .133, .018, .113, .029, .633, .065),
           ("retrained", .064, .004, .130, .029, .424, .099),
           ("qwen3-0.6b", .262, .064, .117, .025, .672, .070)]}
    scaling = {m: [dict(n_names=k, rank_90=v, max_possible=435, n_items=435)
                   for k, v in zip((30, 60, 90, 120), vals)]
               for m, vals in [("unsup-simcse", (37, 55, 69, 78)),
                               ("bge-base", (26, 46, 62, 75)),
                               ("qwen3-0.6b", (23, 41, 56, 68))]}
    superpos = dict(per_model=sp, scaling=scaling)

    # The blind battery
    h1 = {m: dict(overall=o, ci=[o - .03, o + .03],
                  per_subcase={"ln_passive": pa, "ln_subject/object_swap": sw},
                  holds=bool(0.40 <= o <= 0.60))
          for m, o, pa, sw in [("unsup-simcse", .322, .558, .087),
                               ("sup-simcse", .369, .519, .219),
                               ("bge-base", .321, .501, .141),
                               ("e5-base", .572, .656, .489),
                               ("retrained", .292, .549, .036)]}
    langs = ("en", "de", "es", "fr", "ja", "ko", "zh")
    h2 = {m: dict(per_language=dict(zip(langs, v)),
                  max_above_english=max(v) - v[0], holds=bool(max(v) - v[0] <= 0.05))
          for m, v in [("qwen3-0.6b", (.682, .635, .630, .647, .596, .578, .619)),
                       ("qwen3-4b", (.702, .661, .650, .676, .624, .591, .633)),
                       ("qwen3-8b", (.728, .663, .654, .676, .625, .606, .628))]}
    sets = ("sickr", "sts12", "sts13", "sts14", "sts15", "sts16")
    h3 = {m: dict(excess=dict(zip(sets, v)), n_positive=sum(x > 0 for x in v))
          for m, v in [("unsup-simcse", (.607, .383, .150, .261, .136, .136)),
                       ("sup-simcse", (.411, .325, .157, .261, .081, .140)),
                       ("bge-base", (.512, .390, .279, .322, .149, .174)),
                       ("e5-base", (.558, .371, .235, .296, .174, .173)),
                       ("retrained", (.625, .356, .150, .246, .093, .134)),
                       ("graded", (.527, .301, -.038, .083, -.107, .072))]}
    blind = {
        "H1": dict(prediction="every English encoder between 0.40 and 0.60 AUC",
                   holds=False, per_model=h1),
        "H2": dict(prediction="no language more than 0.05 AUC above English, for each size",
                   holds=True, per_model=h2),
        "H3": dict(prediction="encoders positive on >= 5/6; graded below base on >= 5/6",
                   holds=True, per_model=h3, graded_lower_on=6,
                   cross_encoder=dict(zip(sets, (.062, .247, .362, .280, .404, .362)))),
        "H4": dict(prediction="entity swap above base at every seed, matched control not",
                   holds=False, **{"retrained (base)": [.292, .292, .292],
                                   "entity-swap": [.780, .781, .633],
                                   "matched-control": [.714, .694, .651]}),
        "protocol_commit": "01337a7", "code_commit": "341a4b1"}

    ds = ("MSMARCO", "NQ", "NFCorpus", "SciFact", "Quora", "FiQA")
    cost = {c: dict(datasets=dict(zip(ds, v)), mean=mn, change=ch, fooled=fl)
            for c, v, mn, ch, fl in [
                ("retrained (baseline)", (.320, .290, .187, .431, .870, .204), .384, None, .276),
                ("none", (.300, .273, .153, .345, .874, .218), .361, -.060, .280),
                ("random", (.297, .269, .164, .347, .873, .221), .362, -.057, .280),
                ("topic-matched", (.296, .265, .158, .322, .864, .159), .344, -.103, .270),
                ("word-shuffled", (.189, .169, .097, .180, .802, .101), .256, -.332, .258),
                ("non-role swap", (.211, .161, .094, .159, .774, .085), .247, -.356, .280),
                ("role swap", (.280, .231, .170, .315, .852, .155), .334, -.130, .152),
                ("graded, real", (.317, .304, .144, .263, .864, .218), .351, -.084, .220),
                ("graded, shuffled", (.108, .092, .056, .109, .720, .039), .188, -.511, .220)]}

    # F2, verb contrast. Stored as distances (1 - cosine) with the logged ratio and CI.
    equivalence = {}
    for m, ck, cr, v, lo, hi, pnf, pa in [
            ("unsup-simcse", .9905, .9891, 1.095, 1.059, 1.133, 1.6e-10, 1.6e-10),
            ("sup-simcse", .9894, .9913, 0.869, 0.821, 0.920, 4.9e-23, 9.0e-03),
            ("bge-base", .9901, .9894, 1.037, 0.979, 1.099, 8.9e-08, 8.9e-08),
            ("e5-base", .9885, .9889, 0.879, 0.828, 0.932, 1.1e-20, 4.8e-03),
            ("retrained", .9952, .9949, 1.063, 1.038, 1.090, 9.3e-25, 9.3e-25),
            ("qwen3-0.6b", .9270, .9441, 0.705, 0.666, 0.746, 3.5e-48, 1.0)]:
        equivalence[m] = dict(d_kept=1 - ck, d_reversed=1 - cr, ratio=v, ci90=[lo, hi],
                              p_not_further=pnf, p_tost=pa, role_blind=bool(hi < 1.25))

    # Explanation measures. STS-B is graded, so its values are flat; MRPC is class 1/class 0.
    measures = {
        "stsb": {m: dict(n=300, cosine=c, crosspos=x, percell=pc, overlap_mass=om,
                         mass_vs_score=mv, faithfulness=fa)
                 for m, c, x, pc, om, mv, fa in
                 [("bge-base", .543, .322, .58, .180, .645, .90),
                  ("e5-base", .560, .317, .55, .182, .639, .94),
                  ("sup-simcse", .548, .336, .66, .216, .515, .94),
                  ("unsup-simcse", .546, .364, .79, .159, .534, .95)]},
        "mrpc": {m: dict(n=[200, 200], cosine=[c1, c0], crosspos=[x1, x0, pp],
                         percell=[p1, p0], faithfulness=fa)
                 for m, c1, c0, x1, x0, pp, p1, p0, fa in
                 [("qwen3-0.6b", .850, .779, .211, .297, 9e-06, .08, .14, .94),
                  ("sup-simcse", .864, .787, .274, .429, 1e-08, .38, .63, .98)]}}

    return dict(examples=examples, ratios=ratios, equivalence=equivalence, probe=probe,
                layers=layers, measures=measures, entangle=entangle, superpos=superpos,
                blind=blind, cost=cost)


# =============================================================================

FIG_NEEDS = {
    "fig 1  the argument": ["examples"],
    "fig 2  the finding": ["examples", "ratios"],
    "fig 3  the dissociation": ["probe", "layers"],
    "fig 4  why no edit works": ["entangle", "superpos"],
    "fig 5  the blind battery": ["blind"],
    "tables  main text": ["ratios", "equivalence", "probe", "cost", "precision"],
    "tables  supplementary": ["layers", "measures", "graded", "negatives",
                              "entangle", "superpos", "blind"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="write the logged tables instead of reading results/")
    ap.add_argument("--strict", action="store_true", help="exit 1 if a source is missing")
    ap.add_argument("--contrast", default=PASSIVE,
                    help="which minimal-pairs contrast the ratios come from")
    args = ap.parse_args()

    if args.demo:
        data = demo()
        data["meta"] = dict(source="demo", written=dt.datetime.now().isoformat(timespec="seconds"),
                            files={}, note="LAYOUT ONLY -- figures from this are stamped DEMO")
        print("[demo] numbers copied from the logs, for layout only.")
        present = set(k for k in data if k != "meta")
    else:
        data, files, present = {}, {}, set()
        print("collecting from results/\n")
        for key, (pattern, _) in SOURCES.items():
            # the precision sweep lives in the same glob as the headline ratios
            raw, used = merged(pattern, exclude="minimal_pairs_p_" if key == "ratios" else "")
            if not used:
                print(f"  {key:9s} MISSING            {pattern}")
                data[key] = {} if key != "examples" else []
                continue
            data[key] = NORMALISE[key](raw, args.contrast) if key == "ratios" \
                else NORMALISE[key](raw)
            if key == "ratios":          # two more tables live in the same files
                data["equivalence"] = norm_ratios(raw, EQUIV)
                data["percell"] = norm_percell(raw)
                for extra, label in (("equivalence", "matched twins"),
                                     ("percell", "per-cell evidence")):
                    if data[extra]:
                        present.add(extra)
                        print(f"  {extra[:9]:9s} {len(data[extra]):3d} entries        "
                              f"({label}, same files)")
            n = len(data[key].get("per_model", data[key])) if isinstance(data[key], dict) \
                else len(data[key])
            print(f"  {key:9s} {n:3d} entries        " + ", ".join(used))
            files[key] = used
            if n:
                present.add(key)
        raw_p, used_p = collect_precision()
        data["precision"] = raw_p
        if used_p:
            present.add("precision")
            files["precision"] = used_p
            print(f"  {'precision':9s} {len(raw_p):3d} cells          "
                  f"{len(used_p)} files, keyed by model and precision")

        raw_m, used_m = collect_measures()
        data["measures"] = raw_m
        if used_m:
            present.add("measures")
            files["measures"] = used_m
            print(f"  {'measures':9s} {len(raw_m):3d} datasets       " + ", ".join(used_m))
        else:
            print(f"  {'measures':9s} MISSING            {MEASURES}")
        data["meta"] = dict(source="runs", written=dt.datetime.now().isoformat(timespec="seconds"),
                            files=files, contrast=args.contrast)

    os.makedirs(OUT, exist_ok=True)
    # The demo writes its own file, so a real collection can never be overwritten by one.
    path = os.path.join(OUT, "figdata_demo.json" if args.demo else "figdata.json")
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)

    print("\ncoverage")
    missing_any = False
    for fig, needs in FIG_NEEDS.items():
        gaps = [n for n in needs if n not in present]
        missing_any |= bool(gaps)
        state = "ready" if not gaps else "needs " + ", ".join(gaps)
        print(f"  {fig:28s} {state}")
    print(f"\n[done] wrote {path}")
    if missing_any:
        print("A gap is a run that has not happened, not a figure to invent numbers for.")
    if args.strict and missing_any:
        sys.exit(1)


if __name__ == "__main__":
    main()

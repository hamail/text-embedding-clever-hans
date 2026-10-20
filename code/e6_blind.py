"""
e6_blind.py -- the frozen zero-shot battery. This runs ONCE.

PROTOCOL.md fixed four predictions at commit 01337a7, before any of these fourteen sets
was touched by a model. This script tests them and nothing else. It records the commit of
the analysis code beside the protocol commit, so that nobody has to take on trust that the
code did not change after the results were seen, and it refuses to overwrite a result file
that already exists.

  H1  HANS. The five English contrastive encoders separate entailment from non-entailment
      at AUC 0.40-0.60, that is, not at all, on subcases whose only cue is word order.
  H2  PAWS-X. For each Qwen3 size, no language exceeds English by more than 0.05 AUC:
      the deficit is not an artefact of English.
  H3  SICK-R and STS 2012-2016. The contrastive encoders show a positive overlap excess
      above the cross-encoder on at least five of the six, and the graded model sits below
      the base model on at least five of six.
  H4  HANS transfer. Entity-swap fine-tuning raises HANS AUC above the retrained model at
      every seed; the edit-matched control does not.

Usage (from ~/CHVLM/revised):
  python e6_blind.py run                       # all four
  python e6_blind.py run --hypotheses H1 H3    # a subset
  python e6_blind.py run --force               # overwrite an existing result file
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
import time
import sys

import numpy as np

import chvlm

OUT = "results/blind"
PROTOCOL_COMMIT = "01337a7"

ENGLISH_ENCODERS = ["unsup-simcse", "sup-simcse", "bge-base", "e5-base", "retrained"]
STS_SETS = ["sickr", "sts12", "sts13", "sts14", "sts15", "sts16"]
PAWSX_LANGS = ["en", "de", "es", "fr", "ja", "ko", "zh"]
CROSS_ENCODER = "cross-encoder/stsb-roberta-base"


# =============================================================================
# Reading the frozen files without assuming their field names
# =============================================================================

def fields(rows: list[dict]) -> tuple[str, str]:
    """The two sentence fields, whatever the preparation script called them."""
    r = rows[0]
    for a, b in (("s1", "s2"), ("sentence1", "sentence2"), ("premise", "hypothesis"),
                 ("text1", "text2"), ("query", "passage")):
        if a in r and b in r:
            return a, b
    raise SystemExit(f"cannot find the sentence fields in {sorted(r)}")


def labels(rows: list[dict]) -> np.ndarray:
    """Binary labels, 1 meaning entailment or paraphrase."""
    for k in ("label", "gold", "is_paraphrase", "entailment"):
        if k in rows[0]:
            v = [r[k] for r in rows]
            if isinstance(v[0], str):
                return np.array([1 if x.lower() in ("entailment", "paraphrase", "1", "true") else 0
                                 for x in v])
            return np.array([int(x) for x in v])
    raise SystemExit(f"cannot find a label field in {sorted(rows[0])}")


def scores(rows: list[dict]) -> np.ndarray:
    """Graded human scores."""
    for k in ("score", "label", "similarity", "gold"):
        if k in rows[0] and isinstance(rows[0][k], (int, float)):
            return np.array([float(r[k]) for r in rows])
    raise SystemExit(f"cannot find a graded score field in {sorted(rows[0])}")


def similarity(m: chvlm.Loaded, rows: list[dict], bs: int = 64) -> np.ndarray:
    """Centred cosine for every pair of a blind set."""
    a, b = fields(rows)
    E1 = chvlm.embed(m, [r[a] for r in rows], bs=bs, max_length=128)
    E2 = chvlm.embed(m, [r[b] for r in rows], bs=bs, max_length=128)
    return chvlm.cosine(E1, E2, np.vstack([E1, E2]).mean(0))


def cross_encoder_scores(rows: list[dict], bs: int = 64) -> np.ndarray:
    """The joint scorer used as the reference for the overlap excess."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CROSS_ENCODER)
    mdl = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER).to(chvlm.DEVICE).eval()
    a, b = fields(rows)
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            chunk = rows[s:s + bs]
            enc = tok([r[a] for r in chunk], [r[b] for r in chunk], padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(chvlm.DEVICE)
            out.append(mdl(**enc).logits[:, 0].float().cpu().numpy())
    del mdl
    return np.concatenate(out)


# =============================================================================
# H1  HANS: can the similarity tell entailment from its reversal?
# =============================================================================

def hypothesis_1(models: list[str], bs: int) -> dict:
    """HANS subcases are single-class by construction: le_* are entailment, ln_* are not.
    Each non-entailment subcase is therefore scored against the entailment pool, which is
    the comparison the hypothesis is about: can the similarity rank a true entailment above
    a reversal that shares its words."""
    rows = chvlm.load_blind("hans", PROTOCOL_COMMIT)
    y = labels(rows)
    sub = np.array([str(r.get("subcase", r.get("heuristic", "all"))) for r in rows])
    neg_subcases = sorted(set(sub[y == 0]))
    out = {}
    print(f"\nH1  HANS, {len(rows)} pairs ({int((y == 1).sum())} entailment). "
          f"Predicted: every encoder between 0.40 and 0.60.\n")
    print(f"{'model':16s} {'overall':>9s} " + " ".join(f"{c[:20]:>21s}" for c in neg_subcases))
    for name in models:
        m = chvlm.load(name, eager=False)
        sim = similarity(m, rows, bs)
        del m
        pos, neg = sim[y == 1], sim[y == 0]
        per = {c: chvlm.auc(pos, sim[(sub == c) & (y == 0)]) for c in neg_subcases}
        overall = chvlm.auc(pos, neg)
        lo, hi = chvlm.auc_ci(pos, neg, reps=400)
        out[name] = dict(overall=overall, ci=[lo, hi], per_subcase=per,
                         holds=bool(0.40 <= overall <= 0.60))
        print(f"{name:16s} {overall:9.3f} " + " ".join(f"{per[c]:21.3f}" for c in neg_subcases))
    n_ok = sum(v["holds"] for v in out.values())
    print(f"\nH1: {n_ok} of {len(out)} encoders inside 0.40-0.60 "
          f"-> {'HOLDS' if n_ok == len(out) else 'DOES NOT HOLD'}")
    return dict(prediction="every English encoder between 0.40 and 0.60 AUC",
                holds=bool(n_ok == len(out)), per_model=out)


# =============================================================================

def hypothesis_2(models: list[str], bs: int) -> dict:
    out = {}
    print(f"\nH2  PAWS-X, {len(PAWSX_LANGS)} languages. Predicted: no language more than "
          f"0.05 AUC above English.\n")
    print(f"{'model':16s} " + " ".join(f"{l:>7s}" for l in PAWSX_LANGS) + f" {'max - en':>10s}")
    for name in models:
        m = chvlm.load(name, eager=False)
        per = {}
        for lang in PAWSX_LANGS:
            rows = chvlm.load_blind(f"pawsx_{lang}", PROTOCOL_COMMIT)
            y, sim = labels(rows), similarity(m, rows, bs)
            per[lang] = chvlm.auc(sim[y == 1], sim[y == 0])
        del m
        gap = max(per[l] - per["en"] for l in PAWSX_LANGS if l != "en")
        out[name] = dict(per_language=per, max_above_english=gap, holds=bool(gap <= 0.05))
        print(f"{name:16s} " + " ".join(f"{per[l]:7.3f}" for l in PAWSX_LANGS) + f" {gap:+10.3f}")
    n_ok = sum(v["holds"] for v in out.values())
    print(f"\nH2: {n_ok} of {len(out)} models with every language within 0.05 of English "
          f"-> {'HOLDS' if n_ok == len(out) else 'DOES NOT HOLD'}")
    return dict(prediction="no language more than 0.05 AUC above English, for each size",
                holds=bool(n_ok == len(out)), per_model=out)


# =============================================================================
# H3  SICK-R and STS 2012-2016: does the benchmark excess generalise?
# =============================================================================

def hypothesis_3(models: list[str], graded: str, base: str, bs: int) -> dict:
    """Overlap excess above a cross-encoder on six benchmarks never used in development.
    `graded` is a checkpoint path; `base` is the model it was tuned from."""
    data, ref = {}, {}
    for name in STS_SETS:
        rows = chvlm.load_blind(name, PROTOCOL_COMMIT)
        a, b = fields(rows)
        data[name] = dict(rows=rows, gold=scores(rows),
                          overlap=np.array([chvlm.jaccard(r[a], r[b]) for r in rows]))
        print(f"[cross-encoder] {name}: {len(rows)} pairs", flush=True)
        ce = cross_encoder_scores(rows, bs)
        ref[name] = chvlm.partial_spearman(ce, data[name]["overlap"], data[name]["gold"])
    print(f"\nH3  Overlap excess above the cross-encoder. Predicted: contrastive encoders "
          f"positive on at least 5 of 6;\n    the graded model below {base} on at least 5 of 6.\n")
    print(f"{'model':24s} " + " ".join(f"{x:>8s}" for x in STS_SETS) + f" {'positive':>10s}")
    print(f"{'cross-encoder':24s} " + " ".join(f"{ref[x]:8.3f}" for x in STS_SETS) + "  reference")
    out = {}
    for name in list(dict.fromkeys(models + [base, graded])):
        spec = chvlm.MODELS[name] if name in chvlm.MODELS else chvlm.ModelSpec(name, "encoder", "cls")
        m = chvlm.load(spec, eager=False)
        per = {}
        for x in STS_SETS:
            sim = similarity(m, data[x]["rows"], bs)
            per[x] = chvlm.partial_spearman(sim, data[x]["overlap"], data[x]["gold"]) - ref[x]
        del m
        out[name] = dict(excess=per, n_positive=int(sum(v > 0 for v in per.values())))
        print(f"{name[-24:]:24s} " + " ".join(f"{per[x]:+8.3f}" for x in STS_SETS)
              + f" {out[name]['n_positive']:7d}/6", flush=True)
    n_lower = sum(out[graded]["excess"][x] < out[base]["excess"][x] for x in STS_SETS)
    part_a = all(out[n]["n_positive"] >= 5 for n in models)
    print(f"\nH3a: contrastive encoders positive on >= 5 of 6 -> "
          f"{'HOLDS' if part_a else 'DOES NOT HOLD'}")
    print(f"H3b: graded below {base} on {n_lower} of 6 -> "
          f"{'HOLDS' if n_lower >= 5 else 'DOES NOT HOLD'}")
    return dict(prediction="encoders positive on >= 5/6; graded below base on >= 5/6",
                holds=bool(part_a and n_lower >= 5), cross_encoder=ref, per_model=out,
                graded_lower_on=int(n_lower))


# =============================================================================

def hypothesis_4(seeds: list[int], entity: str, control: str, base_name: str, bs: int) -> dict:
    """Does the training fix reach a benchmark it never saw? Checkpoint patterns take
    {seed}, so this does not depend on any earlier naming scheme."""
    rows = chvlm.load_blind("hans", PROTOCOL_COMMIT)
    y = labels(rows)

    def auc_of(ref):
        spec = chvlm.MODELS[ref] if ref in chvlm.MODELS else chvlm.ModelSpec(ref, "encoder", "cls")
        m = chvlm.load(spec, eager=False)
        sim = similarity(m, rows, bs)
        del m
        return chvlm.auc(sim[y == 1], sim[y == 0])

    base = auc_of(base_name)
    print(f"\nH4  HANS transfer. Predicted: entity swap above the base at every seed, the "
          f"matched control not.\n")
    print(f"{'condition':20s} " + " ".join(f"{'seed ' + str(s):>9s}" for s in seeds))
    print(f"{base_name[-20:]:20s} " + " ".join(f"{base:9.3f}" for _ in seeds))
    out = {"base": base, "base_name": base_name}
    for label, pattern in (("entity-swap", entity), ("matched-control", control)):
        vals = []
        for s in seeds:
            try:
                vals.append(auc_of(pattern.format(seed=s)))
            except Exception as e:                        # a missing checkpoint is not a result
                print(f"  [skip] {label} seed {s}: {type(e).__name__}")
                vals.append(float("nan"))
        out[label] = vals
        print(f"{label:20s} " + " ".join(f"{v:9.3f}" for v in vals), flush=True)
    ent = [v for v in out["entity-swap"] if np.isfinite(v)]
    ctl = [v for v in out["matched-control"] if np.isfinite(v)]
    ent_ok = bool(ent) and all(v > base for v in ent)
    ctl_ok = bool(ctl) and all(v > base for v in ctl)
    print(f"\nH4: entity swap above base at every seed: {ent_ok}; "
          f"control above base at every seed: {ctl_ok} "
          f"-> {'HOLDS' if ent_ok and not ctl_ok else 'DOES NOT HOLD'}")
    return dict(prediction="entity swap above base at every seed, matched control not",
                holds=bool(ent_ok and not ctl_ok), **out)


# =============================================================================

def code_commit() -> str:
    """The analysis code's commit, recorded beside the protocol commit so that the code
    cannot be said to have changed after the blind results were seen. Only Python files
    count towards a dirty tree: a run log or a results file written during the run says
    nothing about whether the analysis changed."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py"],
                               capture_output=True, text=True).stdout.strip()
        return h + (" (UNCOMMITTED CHANGES PRESENT)" if dirty else "")
    except Exception:
        return "unavailable"


def cmd_run(args) -> None:
    path = os.path.join(OUT, "blind_results.json")
    if os.path.exists(path) and not args.force:
        sys.exit(f"{path} exists. The battery is meant to run once: inspect it, or pass --force.")
    commit = code_commit()
    if "UNCOMMITTED" in commit and not args.force:
        sys.exit("commit the analysis code first, so its state is recorded with the results "
                 "(or pass --force and say so in the paper).")
    print(f"[blind] protocol {PROTOCOL_COMMIT}, analysis code {commit}")
    print(f"[blind] hypotheses: {', '.join(args.hypotheses)}")

    # A partial rerun must not delete the hypotheses it did not run.  The battery writes
    # one file, so dumping a fresh dict over it loses H2, H3 and H4 when only H1 is rerun
    # -- which is how this file lost three of its four hypotheses once already.  Copy the
    # old file aside, then carry the untouched hypotheses forward.
    result = {"protocol_commit": PROTOCOL_COMMIT, "code_commit": commit, "args": vars(args)}
    if os.path.exists(path):
        stamp = time.strftime("%Y%m%dT%H%M%S")
        bak = path[:-5] + f".{stamp}.bak.json"
        shutil.copy2(path, bak)
        print(f"[blind] previous results copied to {bak}")
        try:
            prior = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            prior = {}
        kept = [h for h in ("H1", "H2", "H3", "H4")
                if h in prior and h not in args.hypotheses]
        for h in kept:
            result[h] = prior[h]
        if kept:
            print(f"[blind] carried forward from the previous run: {', '.join(kept)}")
            result["carried_forward"] = dict(hypotheses=kept,
                                             code_commit=prior.get("code_commit"))

    if "H1" in args.hypotheses:
        result["H1"] = hypothesis_1(args.encoders, args.bs)
    if "H2" in args.hypotheses:
        result["H2"] = hypothesis_2(args.qwen3, args.bs)
    if "H3" in args.hypotheses:
        result["H3"] = hypothesis_3(args.encoders, args.graded, args.base, args.bs)
    if "H4" in args.hypotheses:
        result["H4"] = hypothesis_4(args.seeds, args.entity_ckpt,
                                    args.control_ckpt, args.base, args.bs)

    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(path, "w"), indent=2)
    print(f"\n{'=' * 64}")
    for h in args.hypotheses:
        if h in result:
            print(f"{h}: {'HOLDS' if result[h]['holds'] else 'DOES NOT HOLD'}   "
                  f"({result[h]['prediction']})")
    print(f"{'=' * 64}\n[done] wrote {path}")
    print("These were fixed before the data was touched. Report them as they came out.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the frozen battery")
    r.add_argument("--hypotheses", nargs="+", default=["H1", "H2", "H3", "H4"],
                   choices=["H1", "H2", "H3", "H4"])
    r.add_argument("--encoders", nargs="+", default=ENGLISH_ENCODERS, choices=list(chvlm.MODELS))
    r.add_argument("--qwen3", nargs="+", default=["qwen3-0.6b", "qwen3-4b", "qwen3-8b"],
                   choices=list(chvlm.MODELS))
    r.add_argument("--base", default="retrained",
                   help="the model the fine-tuned checkpoints were tuned from")
    r.add_argument("--graded", default="ckpt/retrained_stsb_real_s42",
                   help="the graded-supervision checkpoint, for H3b")
    r.add_argument("--entity-ckpt", default="ckpt/retrained_neg_role-swap_s{seed}",
                   help="role-reversal checkpoints for H4; {seed} is filled in")
    r.add_argument("--control-ckpt", default="ckpt/retrained_neg_non-role-swap_s{seed}",
                   help="edit-matched control checkpoints for H4")
    r.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    r.add_argument("--bs", type=int, default=64)
    r.add_argument("--force", action="store_true", help="overwrite, or run with uncommitted code")
    args = ap.parse_args()
    {"run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()

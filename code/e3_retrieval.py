"""
e3_retrieval.py -- what the correction costs ordinary retrieval.

Parts, run as subcommands:
  nanobeir  nDCG@10 on NanoBEIR, the small zero-shot retrieval benchmark, for any set of
            checkpoints. Our own pool of PAWS sentences cannot measure retrieval quality
            (every model scores a perfect top-1 there, because no distractor is topically
            related), so a standard benchmark is needed before any claim about what a
            correction costs. This is also the measure on which structure-targeted
            negatives have been reported to cost 8-40% of retrieval performance, so our
            training conditions and their controls can be priced on the same scale.

Note on prefixes: E5 expects "query: " for queries and "passage: " for documents. The
model registry carries one prefix, so E5 is scored with "query: " on both sides, which
understates it slightly. The checkpoints that matter here carry no prefix.

Usage (from ~/CHVLM/revised):
  python e3_retrieval.py nanobeir --checkpoints retrained ckpt/retrained_neg_role-swap_s42
  python e3_retrieval.py nanobeir --checkpoints retrained --datasets NanoNQ NanoSciFact
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import chvlm

OUT = "results/e3"

DEFAULT = ["NanoMSMARCO", "NanoNQ", "NanoNFCorpus", "NanoSciFact", "NanoQuoraRetrieval",
           "NanoFiQA2018"]


def load_nano(name: str):
    """One NanoBEIR dataset: documents, queries and the relevant pairs. Tries the
    Sentence Transformers packaging first, then the original Zeta Alpha one."""
    from datasets import load_dataset
    try:
        corpus = load_dataset("sentence-transformers/NanoBEIR-en", "corpus", split=name)
        queries = load_dataset("sentence-transformers/NanoBEIR-en", "queries", split=name)
        qrels = load_dataset("sentence-transformers/NanoBEIR-en", "qrels", split=name)
    except Exception:
        hub = f"zeta-alpha-ai/{name}"
        corpus = load_dataset(hub, "corpus", split="train")
        queries = load_dataset(hub, "queries", split="train")
        qrels = load_dataset(hub, "qrels", split="train")
    key = lambda row, *names: next(row[n] for n in names if n in row)
    docs = {str(key(r, "_id", "id", "corpus-id")):
            (r.get("title", "") + " " + r.get("text", "")).strip() for r in corpus}
    qs = {str(key(r, "_id", "id", "query-id")): r["text"] for r in queries}
    rel = {}
    for r in qrels:
        q, d = str(key(r, "query-id", "query_id")), str(key(r, "corpus-id", "corpus_id"))
        rel.setdefault(q, {})[d] = float(r.get("score", 1))
    return docs, qs, rel


def ndcg_at_k(ranked: list[str], gains: dict[str, float], k: int = 10) -> float:
    """nDCG@k with the usual log discount, against the best possible ranking."""
    dcg = sum(gains.get(d, 0.0) / np.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    best = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / best) if best > 0 else 0.0


def evaluate_nano(m: chvlm.Loaded, name: str, bs: int, k: int = 10) -> float:
    """Mean nDCG@k over one dataset's queries."""
    docs, qs, rel = load_nano(name)
    ids = list(docs)
    D = chvlm.embed(m, [docs[i] for i in ids], bs=bs, max_length=256)
    Q = chvlm.embed(m, [qs[i] for i in qs], bs=bs, max_length=256)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    scores = Q @ D.T
    out = []
    for row, q in zip(scores, qs):
        if q not in rel:
            continue
        order = np.argsort(-row)[:k]
        out.append(ndcg_at_k([ids[j] for j in order], rel[q], k))
    return float(np.mean(out))


def cmd_nanobeir(args) -> None:
    result = {}
    print(f"[nanobeir] nDCG@{args.k} on {len(args.datasets)} datasets\n")
    print(f"{'checkpoint':34s} " + " ".join(f"{d.replace('Nano', '')[:9]:>9s}"
                                            for d in args.datasets) + f" {'mean':>7s}")
    for name in args.checkpoints:
        spec = chvlm.MODELS[name] if name in chvlm.MODELS else chvlm.ModelSpec(name, "encoder", "cls")
        m = chvlm.load(spec, eager=False)
        scores = {d: evaluate_nano(m, d, args.bs, args.k) for d in args.datasets}
        del m
        mean = float(np.mean(list(scores.values())))
        result[name] = dict(per_dataset=scores, mean=mean)
        print(f"{name[:34]:34s} " + " ".join(f"{scores[d]:9.3f}" for d in args.datasets)
              + f" {mean:7.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "nanobeir.json")
    json.dump(dict(args=vars(args), results=result), open(path, "w"), indent=2)
    if len(result) > 1:
        base, *rest = list(result)
        print(f"\nagainst {base}:")
        for r in rest:
            d = result[r]["mean"] - result[base]["mean"]
            print(f"  {r[:44]:44s} {d:+.3f}  ({100 * d / result[base]['mean']:+.1f}%)")
    print(f"\n[done] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nanobeir", help="zero-shot retrieval quality, before and after training")
    n.add_argument("--checkpoints", nargs="+", required=True,
                   help="the first one is treated as the baseline for the comparison")
    n.add_argument("--datasets", nargs="+", default=DEFAULT)
    n.add_argument("--k", type=int, default=10)
    n.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    {"nanobeir": cmd_nanobeir}[args.cmd](args)


if __name__ == "__main__":
    main()

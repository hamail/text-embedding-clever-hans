"""
train.py -- training interventions on the retrained encoder.

Parts, run as subcommands:
  evaluate Score saved checkpoints on ordinary retrieval as well as on swaps. A model
           whose fooled rate falls only because it has become a worse retriever has not
           been repaired, so top-1 accuracy without any swap in the pool (r = 0) must be
           read beside the fooled rate.

  negatives Why does contrastive training produce overlap-dominated similarity? In-batch
           negatives are random sentences, which share almost no words with the anchor, so
           matching vocabulary already solves the training task and nothing ever requires
           the score to carry who did what. This varies exactly that: the vocabulary the
           added negative shares with its anchor. Six conditions on the same anchors with
           the same schedule; the first three raise the overlap, the last three hold it at
           one and vary what changed instead.

  graded   Fine-tune on STS-B with cosine regression, and its control with the same
           pairs, steps and schedule but the scores shuffled. Graded supervision
           changes many things at once; the shuffled control isolates the one that
           matters, the similarity signal itself. It reports, for every checkpoint,
           the two numbers the claim rests on: how much the model's similarity still
           rewards shared words beyond the human scores, and the retrieval fooled rate.

Later parts will add: mine (swap negatives) and finetune (the contrastive conditions).

Usage (from ~/CHVLM/revised):
  python train.py graded                       # real and shuffled, three seeds each
  python train.py graded --seeds 42 --conditions real
  python train.py evaluate --checkpoints retrained ckpt/retrained_stsb_real_s42 ...
  python train.py negatives --seeds 42            # one seed first, about 40 minutes
"""

from __future__ import annotations

import argparse
import json
import math
import os

import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

import chvlm

OUT = "results/train"


def stsb_train(shuffle: bool, seed: int, exclude: set[tuple[str, str]]):
    """STS-B training pairs, scores scaled to [0, 1]; the control shuffles the scores
    across pairs, so the data and the number of updates are unchanged."""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "stsb", split="train")
    pairs = [(r["sentence1"], r["sentence2"], r["label"] / 5.0) for r in ds
             if (r["sentence1"], r["sentence2"]) not in exclude]
    if shuffle:
        scores = [p[2] for p in pairs]
        np.random.default_rng(seed).shuffle(scores)
        pairs = [(a, b, s) for (a, b, _), s in zip(pairs, scores)]
    return pairs


def fine_tune(base: str, pairs, epochs: int, bs: int, lr: float, seed: int, out_dir: str):
    """Cosine regression on frozen-free weights: the standard sentence-embedding recipe,
    with a fixed number of epochs and no checkpoint selection."""
    torch.manual_seed(seed)
    m = chvlm.load(base, eager=False)
    for p in m.model.parameters():
        p.requires_grad_(True)
    m.model.train()
    opt = torch.optim.AdamW(m.model.parameters(), lr=lr)
    steps = epochs * math.ceil(len(pairs) / bs)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    rng = np.random.default_rng(seed)
    dev = next(m.model.parameters()).device
    step = 0
    for ep in range(epochs):
        order = rng.permutation(len(pairs))                  # a fresh order each epoch
        for i in range(0, len(order), bs):
            batch = [pairs[k] for k in order[i:i + bs]]
            enc = [m.tok([p[j] for p in batch], padding=True, truncation=True, max_length=64,
                         return_tensors="pt").to(dev) for j in (0, 1)]
            e = [chvlm._pool(m.model(**enc[j]).last_hidden_state, enc[j]["attention_mask"],
                             m.spec.pooling, m.model) for j in (0, 1)]
            y = torch.tensor([p[2] for p in batch], device=dev, dtype=e[0].dtype)
            loss = F.mse_loss(F.cosine_similarity(e[0], e[1]), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            if step % 200 == 0:
                print(f"    epoch {ep + 1} step {step}/{steps} loss {loss.item():.4f}", flush=True)
    m.model.eval()
    os.makedirs(out_dir, exist_ok=True)
    m.model.save_pretrained(out_dir); m.tok.save_pretrained(out_dir)
    return m


def evaluate(m: chvlm.Loaded, triplets, sts_pairs) -> dict:
    """The two numbers the causal claim rests on, for one checkpoint."""
    s1 = [r["s1"] for r in sts_pairs]; s2 = [r["s2"] for r in sts_pairs]
    gold = np.array([r["score"] for r in sts_pairs])
    E1, E2 = chvlm.embed(m, s1), chvlm.embed(m, s2)
    mu = np.vstack([E1, E2]).mean(0)
    sim = chvlm.cosine(E1, E2, mu)
    overlap = np.array([chvlm.jaccard(a, b) for a, b in zip(s1, s2)])
    from scipy.stats import spearmanr
    q = [t["query"] for t in triplets]; g = [t["gold"] for t in triplets]; h = [t["hardneg"] for t in triplets]
    Eq, Eg, Eh = chvlm.embed(m, q), chvlm.embed(m, g), chvlm.embed(m, h)
    mu2 = np.vstack([Eq, Eg, Eh]).mean(0)
    fooled = chvlm.cosine(Eq, Eh, mu2) > chvlm.cosine(Eq, Eg, mu2)
    return dict(stsb_rho=float(spearmanr(sim, gold)[0]),
                overlap_excess=chvlm.partial_spearman(sim, overlap, gold),
                fooled=float(fooled.mean()), fooled_flags=fooled.tolist())


def retrieval(m: chvlm.Loaded, triplets, pool: list[str], k_negs: int, seed: int) -> dict:
    """Ordinary retrieval and retrieval under the swap, in one pool.
      top1_clean  the true paraphrase against k random sentences, no swap present
      top1_swap   the same pool with the swapped counterpart added
      fooled      the swap ranked above the true paraphrase"""
    q = [t["query"] for t in triplets]; g = [t["gold"] for t in triplets]; h = [t["hardneg"] for t in triplets]
    Eq, Eg, Eh, Ep = (chvlm.embed(m, x) for x in (q, g, h, pool))
    mu = np.vstack([Eq, Eg, Ep[:1000]]).mean(0)
    nrm = lambda E: (E - mu) / np.linalg.norm(E - mu, axis=1, keepdims=True)
    Q, G, H, P = map(nrm, (Eq, Eg, Eh, Ep))
    rng = np.random.default_rng(seed)
    s_gold = (Q * G).sum(1); s_hard = (Q * H).sum(1); s_pool = Q @ P.T
    clean, swap = [], []
    for i in range(len(q)):
        negs = s_pool[i, rng.choice(len(pool), size=k_negs, replace=False)]
        clean.append(float((negs > s_gold[i]).sum() == 0))
        swap.append(float((negs > s_gold[i]).sum() == 0 and s_hard[i] <= s_gold[i]))
    return dict(top1_clean=float(np.mean(clean)), top1_swap=float(np.mean(swap)),
                fooled=float((s_hard > s_gold).mean()))


def cmd_evaluate(args) -> None:
    from datasets import load_dataset
    triplets = json.load(open(args.triplets))
    pool = [r["sentence2"] for r in load_dataset("google-research-datasets/paws", "labeled_final",
                                                 split="train").select(range(4000))]
    sts_pairs = chvlm.load_pairs("stsb", 300)
    out = {}
    print(f"{'checkpoint':34s} {'top-1 clean':>11s} {'top-1 with swap':>16s} {'fooled':>8s}"
          f" {'STS-B':>7s} {'excess':>8s}")
    for name in args.checkpoints:
        spec = chvlm.MODELS[name] if name in chvlm.MODELS else chvlm.ModelSpec(name, "encoder", "cls")
        m = chvlm.load(spec, eager=False)
        r = retrieval(m, triplets, pool, args.k_negs, args.seed)
        r.update({k: v for k, v in evaluate(m, triplets, sts_pairs).items() if k != "fooled_flags"})
        del m
        out[name] = r
        print(f"{name[:34]:34s} {r['top1_clean']:11.3f} {r['top1_swap']:16.3f} {r['fooled']:8.3f}"
              f" {r['stsb_rho']:7.3f} {r['overlap_excess']:+8.3f}")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "evaluate.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\n[done] wrote {path}")
    print("reading: a fooled rate that falls while top-1 clean also falls means a worse retriever,"
          " not a repaired one")


# =============================================================================
# Negatives at controlled vocabulary overlap
# =============================================================================

KINDS = ["none", "random", "topic", "shuffled", "non-role swap", "role swap"]


def build_index(rows):
    """Content-word sets, and a word -> rows index, for topic matching. Very common words
    are dropped from the index: they add candidates without adding topical similarity."""
    sets = [chvlm.content_words(r["sent"]) for r in rows]
    index = {}
    for i, ws in enumerate(sets):
        for w in ws:
            index.setdefault(w, []).append(i)
    common = {w for w, ids in index.items() if len(ids) > len(rows) / 50}
    return {w: ids for w, ids in index.items() if w not in common}, sets


def make_negative(kind, i, rows, index, sets, rng):
    """One negative for anchor i, of the requested kind."""
    anchor = rows[i]["sent"]
    if kind == "none":
        return None
    if kind == "random":
        return rows[rng.integers(len(rows))]["sent"]
    if kind == "shuffled":                                  # identical vocabulary, order destroyed
        toks = re.findall(r"\w+|[^\w\s]", anchor)
        words = [t for t in toks if t.isalnum()]
        order = list(rng.permutation(len(words)))
        it = iter([words[k] for k in order])
        return "".join((next(it) if t.isalnum() else t) + (" " if n < len(toks) - 1 else "")
                       for n, t in enumerate(toks))
    if kind == "non-role swap":
        return rows[i].get("neg_control")
    if kind == "role swap":
        return rows[i].get("neg_entity")
    shared = Counter(j for w in sets[i] for j in index.get(w, ())[:400] if j != i)
    if not shared:
        return rows[rng.integers(len(rows))]["sent"]
    a = sets[i]                                              # overlap only for the best few
    scored = sorted(((len(a & sets[j]) / max(len(a | sets[j]), 1), j)
                     for j, _ in shared.most_common(50)), reverse=True)
    ok = [j for v, j in scored if 0.15 <= v <= 0.6]          # a genuine middle rung is rare in
    return rows[(ok[0] if ok else scored[0][1])]["sent"]     # Wikipedia; else take the closest


def contrastive_finetune(base, texts, negs, steps, bs, lr, temp, seed, out_dir):
    """SimCSE fine-tuning: two dropout views as the positive pair, the batch as negatives,
    plus the supplied negative when there is one."""
    torch.manual_seed(seed)
    m = chvlm.load(base, eager=False)
    for p in m.model.parameters():
        p.requires_grad_(True)
    m.model.train()
    opt = torch.optim.AdamW(m.model.parameters(), lr=lr)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    rng = np.random.default_rng(seed)
    dev = next(m.model.parameters()).device
    order = rng.permutation(len(texts))
    for step in range(steps):
        idx = order[(step * bs) % max(len(order) - bs, 1):][:bs]
        batch = [texts[k] for k in idx]
        enc = m.tok(batch, padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
        z = [chvlm._pool(m.model(**enc).last_hidden_state, enc["attention_mask"], m.spec.pooling,
                         m.model) for _ in range(2)]                     # two dropout views
        z = [torch.nn.functional.normalize(x, dim=-1) for x in z]
        logits = z[0] @ z[1].T / temp
        have = [k for k in idx if negs and negs[k]]
        if have:                                                        # one extra column per anchor
            ne = m.tok([negs[k] for k in idx], padding=True, truncation=True, max_length=32,
                       return_tensors="pt").to(dev)
            zn = torch.nn.functional.normalize(
                chvlm._pool(m.model(**ne).last_hidden_state, ne["attention_mask"], m.spec.pooling,
                            m.model), dim=-1)
            logits = torch.cat([logits, ((z[0] * zn).sum(-1) / temp).unsqueeze(1)], dim=1)
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch), device=dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step + 1}/{steps} loss {loss.item():.4f}", flush=True)
    m.model.eval()
    os.makedirs(out_dir, exist_ok=True)
    m.model.save_pretrained(out_dir); m.tok.save_pretrained(out_dir)
    return m


def cmd_negatives(args) -> None:
    rows = [json.loads(l) for l in open(args.negatives)]
    random.Random(args.seed).shuffle(rows)
    rows = [r for r in rows if r.get("neg_entity") and r.get("neg_control")][:args.n_anchors]
    index, sets = build_index(rows)
    texts = [r["sent"] for r in rows]
    sts_pairs = chvlm.load_pairs("stsb", 300)
    triplets = json.load(open(args.triplets))
    base = chvlm.load(args.base, eager=False)
    before = evaluate(base, triplets, sts_pairs); del base
    print(f"[{args.base}] STS-B rho {before['stsb_rho']:.3f}   overlap excess "
          f"{before['overlap_excess']:+.3f}   fooled {before['fooled']:.3f}")
    result = {"before": {k: v for k, v in before.items() if k != "fooled_flags"}, "after": {}}
    print(f"\n{'negative':16s} {'its overlap':>11s} {'STS-B':>7s} {'overlap excess':>15s} {'fooled':>8s}")
    for kind in args.kinds:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            negs = [make_negative(kind, i, rows, index, sets, rng) for i in range(len(rows))]
            ov = float(np.mean([chvlm.jaccard(t, n) for t, n in zip(texts, negs) if n])) if any(negs) else 0.0
            tag = f"{args.base}_neg_{kind.replace(' ', '-')}_s{seed}"
            m = contrastive_finetune(args.base, texts, negs, args.steps, args.bs, args.lr,
                                     args.temp, seed, f"ckpt/{tag}")
            after = evaluate(m, triplets, sts_pairs); del m
            result["after"][tag] = {**{k: v for k, v in after.items() if k != "fooled_flags"},
                                    "kind": kind, "seed": seed, "negative_overlap": ov}
            print(f"{kind:16s} {ov:11.3f} {after['stsb_rho']:7.3f} {after['overlap_excess']:+15.3f} "
                  f"{after['fooled']:8.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "negatives.json")
    json.dump(dict(args=vars(args), **result), open(path, "w"), indent=2)
    print(f"\n[done] wrote {path}")
    print("reading: if word matching survives because nothing in training punishes it, the overlap"
          " excess and the fooled rate should fall as the negative's vocabulary overlap rises")


def cmd_graded(args) -> None:
    sts_pairs = chvlm.load_pairs("stsb", 300)
    triplets = json.load(open(args.triplets))
    exclude = {(r["s1"], r["s2"]) for r in sts_pairs}
    base = chvlm.load(args.base, eager=False)
    before = evaluate(base, triplets, sts_pairs)
    del base
    print(f"[{args.base}] STS-B rho {before['stsb_rho']:.3f}   overlap excess "
          f"{before['overlap_excess']:+.3f}   fooled {before['fooled']:.3f}")

    result = {"before": {k: v for k, v in before.items() if k != "fooled_flags"}, "after": {}}
    for cond in args.conditions:
        for seed in args.seeds:
            tag = f"{args.base}_stsb_{cond}_s{seed}"
            print(f"\n[train] {cond} scores, seed {seed} -> ckpt/{tag}", flush=True)
            pairs = stsb_train(cond == "shuffled", seed, exclude)
            if args.max_train:
                pairs = pairs[:args.max_train]
            m = fine_tune(args.base, pairs, args.epochs, args.bs, args.lr, seed, f"ckpt/{tag}")
            after = evaluate(m, triplets, sts_pairs); del m
            fixed, broke, p = chvlm.mcnemar(np.array(before["fooled_flags"]), np.array(after["fooled_flags"]))
            result["after"][tag] = {**{k: v for k, v in after.items() if k != "fooled_flags"},
                                    "mcnemar": dict(fixed=fixed, broke=broke, p=p)}
            print(f"    STS-B rho {after['stsb_rho']:.3f}   overlap excess {after['overlap_excess']:+.3f}"
                  f"   fooled {after['fooled']:.3f}   queries fixed {fixed}, broken {broke}, P={p:.1e}")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "graded.json")
    json.dump(dict(args=vars(args), **result), open(path, "w"), indent=2)
    print(f"\n[done] wrote {path}")
    print("reading: real scores should lower the overlap excess and the fooled rate; shuffled scores"
          " use the same pairs, steps and schedule, so whatever they change is the training itself")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("graded", help="graded supervision and its shuffled-score control")
    g.add_argument("--base", default="retrained", choices=list(chvlm.MODELS))
    g.add_argument("--conditions", nargs="+", default=["real", "shuffled"], choices=["real", "shuffled"])
    g.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    g.add_argument("--epochs", type=int, default=4)
    g.add_argument("--bs", type=int, default=16)
    g.add_argument("--lr", type=float, default=2e-5)
    g.add_argument("--max-train", type=int, default=0)       # 0 = the whole split
    g.add_argument("--triplets", default="results/phase2_retrieval/unsup-simcse-bert-base-uncased/triplets.json")
    n = sub.add_parser("negatives", help="vary how much vocabulary the negative shares")
    n.add_argument("--base", default="retrained", choices=list(chvlm.MODELS))
    n.add_argument("--kinds", nargs="+", default=KINDS, choices=KINDS)
    n.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    n.add_argument("--negatives", default="negatives.jsonl")
    n.add_argument("--n-anchors", type=int, default=20000)
    n.add_argument("--steps", type=int, default=200)
    n.add_argument("--bs", type=int, default=64)
    n.add_argument("--lr", type=float, default=1e-5)
    n.add_argument("--temp", type=float, default=0.05)
    n.add_argument("--seed", type=int, default=7)
    n.add_argument("--triplets", default="results/phase2_retrieval/unsup-simcse-bert-base-uncased/triplets.json")
    e = sub.add_parser("evaluate", help="ordinary retrieval beside the swap, for saved checkpoints")
    e.add_argument("--checkpoints", nargs="+", required=True)
    e.add_argument("--k-negs", type=int, default=99)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--triplets", default="results/phase2_retrieval/unsup-simcse-bert-base-uncased/triplets.json")
    args = ap.parse_args()
    {"graded": cmd_graded, "evaluate": cmd_evaluate, "negatives": cmd_negatives}[args.cmd](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The matched-twin comparison on real text.

SemEval-2010 Task 8 marks two entities in every sentence. Where the label names
a directed relation, exchanging the two spans reverses the claim. Where the
label is Other, the two entities stand in no annotated relation, so the identical
exchange moves the same words without reversing anything.

That is the twins design on sampled rather than constructed text:

                 mean d(s, s') over directed relations
    rho_real  =  -------------------------------------
                 mean d(s, s') over Other

A score that reads roles moves further on the directed set. A score that reads
words moves the same distance on both, and rho_real sits at one.

The swap displaces more tokens when the two spans are far apart or the sentence
is long, so the ratio is also reported within strata of sentence length and span
separation, and as a stratum-weighted average.

Usage
-----
    python semeval_ratio.py --model BAAI/bge-base-en-v1.5
    python semeval_ratio.py --model BAAI/bge-base-en-v1.5 --limit 400
"""

from __future__ import annotations

import argparse, json, os, re, sys
import numpy as np
from scipy import stats

DIRECTED = ["Cause-Effect", "Entity-Destination", "Entity-Origin",
            "Instrument-Agency", "Message-Topic", "Product-Producer"]
E1 = re.compile(r"<e1>(.*?)</e1>", re.S)
E2 = re.compile(r"<e2>(.*?)</e2>", re.S)


def strip_tags(s): return re.sub(r"</?e[12]>", "", s)


def fix_articles(s):
    s = re.sub(r"\ba(?=\s+[aeiouAEIOU])", "an", s)
    s = re.sub(r"\bA(?=\s+[aeiouAEIOU])", "An", s)
    s = re.sub(r"\ban(?=\s+[^aeiouAEIOU\W\d_])", "a", s)
    s = re.sub(r"\bAn(?=\s+[^aeiouAEIOU\W\d_])", "A", s)
    return s


def swap(tagged):
    m1, m2 = E1.search(tagged), E2.search(tagged)
    if not m1 or not m2:
        return None
    a, b = m1.group(1), m2.group(1)
    if a.strip().lower() == b.strip().lower():
        return None
    out = tagged[:m1.start()] + f"<e1>{b}</e1>" + tagged[m1.end():]
    m2b = E2.search(out)
    return out[:m2b.start()] + f"<e2>{a}</e2>" + out[m2b.end():]


def build(rows, names):
    """Both arms, built identically. Only the label decides which arm."""
    out = []
    for r in rows:
        lab = names[r["relation"]]
        kind = "directed" if lab[:-7] in DIRECTED else ("other" if lab == "Other" else None)
        if kind is None:
            continue
        sw = swap(r["sentence"])
        if sw is None:
            continue
        plain = strip_tags(r["sentence"])
        m1, m2 = E1.search(r["sentence"]), E2.search(r["sentence"])
        # how far apart the two spans sit, in words of the untagged sentence
        between = strip_tags(r["sentence"][m1.end():m2.start()])
        out.append(dict(arm=kind, label=lab,
                        sentence=fix_articles(plain),
                        reversed=fix_articles(strip_tags(sw)),
                        n_tok=len(plain.split()),
                        gap=len(between.split())))
    return out


def embed(texts, model_name, batch):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("pip install sentence-transformers, or wire in your own encoder")
    m = SentenceTransformer(model_name, trust_remote_code=True)
    return np.asarray(m.encode(texts, batch_size=batch, show_progress_bar=True,
                               convert_to_numpy=True), dtype=np.float32)


def tost(D, O, delta=1.25):
    """The equivalence test of the paper, applied across the two arms.

    The paper aggregates per-item ratios as a geometric mean and tests log rho
    with a t-interval. Here the two distances come from different sentences, so
    the same quantity is the difference of mean log distances, whose exponential
    is the ratio of geometric means, with a two-sample standard error and
    Welch's degrees of freedom."""
    ld, lo = np.log(D), np.log(O)
    m = ld.mean() - lo.mean()
    vd, vo = ld.var(ddof=1) / ld.size, lo.var(ddof=1) / lo.size
    se = np.sqrt(vd + vo)
    df = (vd + vo) ** 2 / (vd ** 2 / (ld.size - 1) + vo ** 2 / (lo.size - 1))
    lg = np.log(delta)
    p_not_further = stats.t.cdf((m - lg) / se, df)
    p_alike = max(p_not_further, 1 - stats.t.cdf((m + lg) / se, df))
    t = stats.t.ppf(0.95, df)
    return dict(rho_geo=float(np.exp(m)),
                ci90=[float(np.exp(m - t * se)), float(np.exp(m + t * se))],
                p_not_further=float(p_not_further), p_alike=float(p_alike),
                df=float(df),
                equivalent=bool(p_not_further < 0.05 and p_alike < 0.05))


def boot_ratio(x, y, n=2000, seed=0):
    """Percentile interval on mean(x) / mean(y), resampling each arm."""
    rng = np.random.default_rng(seed)
    r = [rng.choice(x, x.size).mean() / rng.choice(y, y.size).mean() for _ in range(n)]
    return float(np.percentile(r, 5)), float(np.percentile(r, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="results/semeval_ratio.json")
    a = ap.parse_args()

    from datasets import load_dataset
    try:
        ds = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8")
    except Exception:
        ds = load_dataset("sem_eval_2010_task_8")
    names = ds["train"].features["relation"].names
    rows = [r for sp in ds for r in ds[sp]]
    if a.limit:
        rows = rows[:a.limit]

    items = build(rows, names)
    arm = np.array([it["arm"] for it in items])
    print(f"[items] directed {int((arm=='directed').sum())}, "
          f"other {int((arm=='other').sum())}")

    X = embed([it["sentence"] for it in items] +
              [it["reversed"] for it in items], a.model, a.batch)
    n = len(items)
    X -= X.mean(0, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    d = 1.0 - np.einsum("ij,ij->i", X[:n], X[n:])          # distance across the swap

    D, O = d[arm == "directed"], d[arm == "other"]
    ratio = D.mean() / O.mean()
    lo, hi = boot_ratio(D, O)
    print(f"\n[{a.model}]")
    print(f"  mean d across the swap, directed relations : {D.mean():.4f}")
    print(f"  mean d across the swap, Other              : {O.mean():.4f}")
    print(f"  rho_real, ratio of means                   : {ratio:.3f}  "
          f"(90% CI {lo:.3f}-{hi:.3f})")
    T = tost(D, O)
    print(f"  rho_real, geometric, as in Supp. Table 1   : {T['rho_geo']:.3f}  "
          f"(90% CI {T['ci90'][0]:.3f}-{T['ci90'][1]:.3f})")
    print(f"    P not further {T['p_not_further']:.2e}   "
          f"P alike {T['p_alike']:.2e}   "
          f"equivalent within [0.8, 1.25]: {'yes' if T['equivalent'] else 'no'}")

    # the swap displaces more when the sentence is long or the spans are far
    # apart, so compare within strata of both and reweight
    tok = np.array([it["n_tok"] for it in items])
    gap = np.array([it["gap"] for it in items])
    tb = np.digitize(tok, np.quantile(tok, [.25, .5, .75]))
    gb = np.digitize(gap, np.quantile(gap, [.25, .5, .75]))
    print(f"\n  within strata of sentence length x span separation")
    num = den = 0.0
    for t in range(4):
        for g in range(4):
            m = (tb == t) & (gb == g)
            dd, oo = d[m & (arm == "directed")], d[m & (arm == "other")]
            if dd.size < 20 or oo.size < 20:
                continue
            num += dd.mean() * dd.size
            den += oo.mean() * dd.size            # weight by the directed count
            print(f"    len q{t+1}, gap q{g+1}: {dd.mean()/oo.mean():>6.3f}"
                  f"   (n = {dd.size}, {oo.size})")
    print(f"  stratum-weighted rho_real                  : {num/den:.3f}"
          if den else "  no stratum had enough of both arms")

    res = dict(model=a.model, n_directed=int(D.size), n_other=int(O.size),
               d_directed=float(D.mean()), d_other=float(O.mean()),
               rho_real=float(ratio), ci90=[lo, hi], tost=T,
               rho_real_stratified=float(num/den) if den else None)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
    prev[a.model] = res
    json.dump(prev, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()

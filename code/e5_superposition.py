"""
e5_superposition.py -- testing a published account of why this must happen.

Kang et al. (2025) prove that a unit-sphere cosine space satisfying basic concept
categorisation forces two-concept embeddings into linear superposition, and that
superposition is commutative, so swapped bindings must embed identically. The proof
reaches "identically" through one assumption: the perturbation an attribute adds is a
generic direction for that attribute, v = p t(a), independent of which object carries it.
The conclusion about cosine does not need that assumption; the conclusion about the
representation does.

This measures the assumption directly, on text. For a sentence and its role reversal,

    S = (e_ab + e_ba) / 2     the commutative part, what superposition predicts
    D = (e_ab - e_ba) / 2     the binding part, which the assumption sets to zero

and asks three questions of D:
  size      how large is it against the content it sits on, and is it small enough to
            explain why cosine cannot see it
  form      is it one generic role axis (the assumption) or one direction per entity
            (our account): alignment between pairs, and the rank needed to hold it
  scaling   does that rank grow as the entity vocabulary grows -- the prediction that
            separates the two accounts, and the reason a low-rank edit cannot work

Run (from ~/CHVLM/revised):
  python e5_superposition.py decompose
  python e5_superposition.py decompose --models unsup-simcse qwen3-0.6b --per-pair 4
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

import chvlm
import e2_mechanism as e2

OUT = "results/e5"


# =============================================================================
# Stimuli: a sentence and its role reversal, sharing every word
# =============================================================================

# Surnames beyond the 40 the probe uses, so the rank of the binding part can be followed
# to a vocabulary large enough to say whether its growth continues or saturates. The first
# 40 are e2.NAMES unchanged, so the smaller vocabularies match the probe's stimuli exactly.
MORE_NAMES = ("Bennett Fletcher Marshall Chapman Hartley Sinclair Whitfield Ashford Bramley "
              "Caldwell Denton Ellery Fairbank Garnett Halstead Ingram Jarrow Kingsley Langton "
              "Merrick Norwood Ockley Pemberton Quayle Radcliffe Selby Thornton Upton Vance "
              "Wexford Yardley Ziegler Ambrose Blackwood Carrington Dunmore Everly Fenwick "
              "Granville Hawthorne Islington Jessop Kendrick Linfield Mowbray Netherton "
              "Oakhurst Prescott Quinton Rutherford Stanhope Tewkesbury Underhill Vernon "
              "Wetherby Yealand Zenobia Aldridge Barrowman Crawford Delaney Emberly Fairweather "
              "Gladstone Harcourt Inglewood Jennings Kirkwood Lockhart Montrose Newbury "
              "Ormsby Pendleton Quillan Ravenscroft Sedgwick Trenholm Ulverston Vandermere "
              "Wolseley Yarrowby").split()


def name_pool(n: int) -> list[str]:
    """The first 40 are the probe's own names, so small vocabularies match those stimuli."""
    pool = list(e2.NAMES) + [x for x in MORE_NAMES if x not in e2.NAMES]
    if n > len(pool):
        raise SystemExit(f"only {len(pool)} names available, asked for {n}")
    return pool[:n]


def swap_items(per_pair: int, seed: int, n_names: int, fixed_frame: bool = False) -> list[dict]:
    """For each unordered name pair and each sampled frame, the sentence with A as agent
    and the same sentence with the names exchanged. Vocabulary is identical; only who
    fills which role differs.

    With fixed_frame, every item uses the same frame, verb, year and city, so the entity
    pair is the ONLY thing that varies. The rank of the binding part is then driven by
    entities alone: anything above the entity count comes from somewhere else."""
    rng = random.Random(seed)
    names = name_pool(n_names)
    const = dict(f=e2.FRAMES[0], v=e2.VERBS[0], year=2011, city=e2.CITIES[0])
    items = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for k in range(1 if fixed_frame else per_pair):
                if fixed_frame:
                    f, v, kw = const["f"], const["v"], dict(year=const["year"], city=const["city"])
                else:
                    f, v = rng.choice(e2.FRAMES), rng.choice(e2.VERBS)
                    kw = dict(year=rng.randint(1995, 2020), city=rng.choice(e2.CITIES))
                items.append(dict(ab=f.format(S=f"{a} {v} {b}", **kw),
                                  ba=f.format(S=f"{b} {v} {a}", **kw),
                                  a=a, b=b, pair=f"{a}|{b}", sample=k))
    return items


# =============================================================================
# The decomposition
# =============================================================================

def decompose(E_ab: np.ndarray, E_ba: np.ndarray, mu: np.ndarray):
    """Commutative part S and binding part D, on centred embeddings."""
    S = (E_ab + E_ba) / 2 - mu
    D = (E_ab - E_ba) / 2
    return S, D


def unit(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def size_of_binding(S: np.ndarray, D: np.ndarray, E_ab: np.ndarray, E_ba: np.ndarray,
                    mu: np.ndarray) -> dict:
    """How large the binding part is, and the similarity it leaves behind."""
    ratio = np.linalg.norm(D, axis=1) / np.maximum(np.linalg.norm(S, axis=1), 1e-12)
    cos = chvlm.cosine(E_ab, E_ba, mu)
    lo, hi = chvlm.ci(chvlm.bootstrap(lambda ix: float(ratio[ix].mean()), len(ratio)))
    return dict(binding_fraction=float(ratio.mean()), binding_fraction_ci=[lo, hi],
                binding_fraction_sd=float(ratio.std()),
                cos_with_reversal=float(cos.mean()),
                variance_share=float((D ** 2).sum() / ((S ** 2).sum() + (D ** 2).sum())))


def form_of_binding(D: np.ndarray, items: list[dict], seed: int = 0) -> dict:
    """One generic role axis, or one direction per entity?
      across_pairs   mean |cos| between the binding directions of different name pairs,
                     after averaging within a pair; a generic axis gives a value near 1
      within_pair    the same within a pair, across frames -- an upper bound set by noise
      random_level   the same for random directions, the floor
      rank_90        principal directions needed for 90% of the binding energy; a generic
                     axis needs 1"""
    by_pair = {}
    for k, it in enumerate(items):
        by_pair.setdefault(it["pair"], []).append(k)
    means = unit(np.array([D[rows].mean(0) for rows in by_pair.values()]))
    off = lambda M: M[~np.eye(len(M), dtype=bool)]
    within = [float(np.abs(unit(D[rows]) @ unit(D[rows]).T)[~np.eye(len(rows), dtype=bool)].mean())
              for rows in by_pair.values() if len(rows) > 1]
    rng = np.random.default_rng(seed)
    G = unit(rng.normal(size=means.shape))
    s = np.linalg.svd(D - D.mean(0), compute_uv=False)
    energy = np.cumsum(s ** 2) / (s ** 2).sum()
    return dict(n_pairs=len(by_pair),
                across_pairs=float(np.abs(off(means @ means.T)).mean()),
                within_pair=float(np.mean(within)) if within else float("nan"),
                random_level=float(np.abs(off(G @ G.T)).mean()),
                rank_90=int(np.searchsorted(energy, 0.90) + 1),
                top1_energy=float(energy[0]))


def entity_alignment(D: np.ndarray, E_ab: np.ndarray, E_ba: np.ndarray, mu: np.ndarray,
                     items: list[dict]) -> dict:
    """Does a pair's binding direction lie along its own entities' identity directions?
    The assumption v = p t(a) predicts no relation: the direction belongs to the role,
    not the entity."""
    ident, rows_of = {}, {}
    for k, it in enumerate(items):
        for n in (it["a"], it["b"]):
            rows_of.setdefault(n, []).extend([k])
    allE = np.vstack([E_ab, E_ba])
    for n, rows in rows_of.items():
        idx = np.array(rows)
        ident[n] = allE[np.concatenate([idx, idx + len(E_ab)])].mean(0) - mu
    names = list(ident)
    I = unit(np.array([ident[n] for n in names]))
    by_pair = {}
    for k, it in enumerate(items):
        by_pair.setdefault(it["pair"], []).append(k)
    own, other = [], []
    for pair, rows in by_pair.items():
        d = unit(D[rows].mean(0)[None])[0]
        a, b = pair.split("|")
        own.append(max(abs(d @ I[names.index(a)]), abs(d @ I[names.index(b)])))
        other += [abs(d @ I[names.index(n)]) for n in names if n not in (a, b)][:8]
    return dict(own_entity=float(np.mean(own)), other_entities=float(np.mean(other)))


def scaling(D: np.ndarray, items: list[dict], seed: int = 0) -> list[dict]:
    """How the rank of the binding part grows with the ENTITY VOCABULARY -- the prediction
    that separates the two accounts. A generic role axis needs one direction however many
    entities are involved; a term bound to each entity needs its own, so the rank climbs
    and no fixed low-rank edit can hold it.

    The number of items is held fixed across vocabulary sizes. Rank is bounded by the item
    count, so drawing more sentences alongside more entities would inflate the rank for a
    reason that has nothing to do with binding."""
    names = sorted({n for it in items for n in (it["a"], it["b"])})
    rng = np.random.default_rng(seed)
    sizes = [max(2, int(round(len(names) * f))) for f in (0.25, 0.5, 0.75, 1.0)]
    pools = {k: [r for r, it in enumerate(items)
                 if it["a"] in set(names[:k]) and it["b"] in set(names[:k])] for k in sizes}
    n_fixed = min(len(p) for p in pools.values())            # what the smallest vocabulary allows
    out = []
    for k in sizes:
        rows = rng.choice(pools[k], size=n_fixed, replace=False)
        M = D[rows] - D[rows].mean(0)
        s = np.linalg.svd(M, compute_uv=False)
        energy = np.cumsum(s ** 2) / (s ** 2).sum()
        out.append(dict(n_names=k, n_items=int(n_fixed), max_possible=int(min(n_fixed, D.shape[1])),
                        rank_90=int(np.searchsorted(energy, 0.90) + 1),
                        top1_energy=float(energy[0])))
    return out


def cmd_decompose(args) -> None:
    items = swap_items(args.per_pair, args.seed, args.n_names, args.fixed_frame)
    print(f"[superposition] {'one frame and verb throughout; ' if args.fixed_frame else ''}"
          f"{len(items)} sentence pairs, "
          f"{len({i['pair'] for i in items})} entity pairs, identical vocabulary within a pair\n")
    result = {}
    print(f"{'model':16s} {'|D|/|S|':>9s} {'var share':>10s} {'cos(ab,ba)':>11s} "
          f"{'across pairs':>13s} {'random':>8s} {'own entity':>11s} {'others':>8s} {'rank 90%':>9s}")
    for name in args.models:
        m = chvlm.load(name, eager=False)
        E_ab = chvlm.embed(m, [it["ab"] for it in items], bs=128)
        E_ba = chvlm.embed(m, [it["ba"] for it in items], bs=128)
        del m
        mu = np.vstack([E_ab, E_ba]).mean(0)
        S, D = decompose(E_ab, E_ba, mu)
        r = {**size_of_binding(S, D, E_ab, E_ba, mu), **form_of_binding(D, items, args.seed),
             **entity_alignment(D, E_ab, E_ba, mu, items)}
        r["scaling"] = scaling(D, items, args.seed)
        result[name] = r
        print(f"{name:16s} {r['binding_fraction']:9.3f} {r['variance_share']:10.3f} "
              f"{r['cos_with_reversal']:11.4f} {r['across_pairs']:13.3f} {r['random_level']:8.3f} "
              f"{r['own_entity']:11.3f} {r['other_entities']:8.3f} {r['rank_90']:9d}", flush=True)

    sc0 = result[args.models[0]]["scaling"]
    print(f"\n[scaling] directions for 90% of the binding part as the vocabulary grows, at a fixed"
          f" {sc0[0]['n_items']} items (rank could reach {sc0[0]['max_possible']} in every column)")
    print(f"{'model':16s} " + " ".join(f"{d['n_names']:>5d} names" for d in sc0))
    for name, r in result.items():
        print(f"{name:16s} " + " ".join(f"{d['rank_90']:>11d}" for d in r["scaling"]))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "superposition.json")
    json.dump(dict(args=vars(args), n_items=len(items), results=result), open(path, "w"), indent=2)
    print(f"\n[done] wrote {path}")
    print("reading: a generic role axis (the published assumption) gives across-pairs alignment near 1"
          " and rank 1, with no relation to the entities. One direction per entity gives alignment near"
          " the random level, own-entity far above other entities, and a rank that climbs with the"
          " vocabulary -- which is also why a low-rank edit cannot work.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decompose", help="commutative and binding parts of a role reversal")
    d.add_argument("--models", nargs="+", default=["unsup-simcse", "sup-simcse", "bge-base",
                                                   "e5-base", "retrained", "qwen3-0.6b"],
                   choices=list(chvlm.MODELS))
    d.add_argument("--per-pair", type=int, default=2)
    d.add_argument("--n-names", type=int, default=40)
    d.add_argument("--seed", type=int, default=11)
    d.add_argument("--fixed-frame", action="store_true",
                   help="one frame and verb throughout, so only the entity pair varies and the"
                        " rank of the binding part is driven by entities alone")
    args = ap.parse_args()
    {"decompose": cmd_decompose}[args.cmd](args)


if __name__ == "__main__":
    main()

"""
e4_intervention.py -- can the cue be removed after training?

Parts, run as subcommands:
  entanglement  Two measurements of why no edit works.
                (a) The exchange rate, from the readout sweep: general similarity lost
                    per unit of swap similarity removed, for role swaps against non-role
                    swaps. If role structure is dearer to separate, that is entanglement
                    priced rather than asserted.
                (b) The direction overlap: for each name, its role direction (that name as
                    agent minus as patient) against the span of word-identity directions
                    built from ordinary Wikipedia words. A role direction lying inside that
                    span cannot be amplified without disturbing word matching; a random
                    direction gives the scale, and removing the span shows how much of
                    benchmark agreement those directions carry.

Later parts will add: masking, pruning and projection, from the same file.

Usage (from ~/CHVLM/revised):
  python e4_intervention.py entanglement
  python e4_intervention.py entanglement --models retrained qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import chvlm
import e2_mechanism as e2

OUT = "results/e4"


# =============================================================================
# (a) The exchange rate, from the readout sweep
# =============================================================================

def exchange_rates(path: str) -> dict:
    """Points of STS-B lost per 0.01 of swap similarity removed, by edit and setting.
    A higher rate means that structure is dearer to separate."""
    data = json.load(open(path))["results"]
    out = {}
    for model, rows in data.items():
        base = next(r for r in rows if r["edit"] == "none")
        for r in rows:
            if r["edit"] == "none":
                continue
            removed = base["train_swap"] - r["train_swap"]          # on its own training swaps
            lost = base["stsb_test"] - r["stsb_test"]
            if removed > 0.002:                                     # ignore edits that did nothing
                out.setdefault(model, []).append(
                    dict(edit=r["edit"], rank=r["rank"], lam=r["lam"], removed=removed,
                         stsb_lost=lost, points_per_001=100 * lost / (removed / 0.01) / 100))
    return out


# =============================================================================
# (b) The direction overlap
# =============================================================================

def directions(E: np.ndarray, items: list[dict]):
    """Role and identity directions, one of each per name.
      role     that name as agent, minus that name as patient
      identity sentences containing that name, minus the overall mean"""
    names = sorted({it["agent"] for it in items})
    mu = E.mean(0)
    role, ident = [], []
    for n in names:
        ag = [i for i, it in enumerate(items) if it["agent"] == n]
        pa = [i for i, it in enumerate(items) if it["patient"] == n]
        if len(ag) < 5 or len(pa) < 5:
            continue
        role.append(E[ag].mean(0) - E[pa].mean(0))
        ident.append(E[ag + pa].mean(0) - mu)
    return np.array(role), np.array(ident), names


def unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def role_axis(R: np.ndarray, I: np.ndarray, rank: int) -> dict:
    """Is there one direction an edit could amplify?
      shared_axis   mean |cos| between the role directions of different names; at the
                    random level, every name has its own and no single direction exists
      own_name      mean |cos| between a name's role direction and its own identity direction
      other_name    the same against other names' identity directions, as a control
      variance_top  share of the role directions' energy in their own leading directions:
                    how few directions would carry them if they did align"""
    Ru, Iu = unit(R), unit(I)
    off_diag = lambda M: M[~np.eye(len(M), dtype=bool)]
    C = np.abs(Ru @ Ru.T)
    own = np.abs((Ru * Iu).sum(1))
    P = np.abs(Ru @ Iu.T)
    rng = np.random.default_rng(0)
    G = unit(rng.normal(size=(len(R), R.shape[1])))
    sv = np.linalg.svd(R, compute_uv=False)
    k = min(rank, len(sv))
    return dict(n_names=len(R), shared_axis=float(off_diag(C).mean()),
                random_level=float(off_diag(np.abs(G @ G.T)).mean()),
                own_name=float(own.mean()), other_name=float(off_diag(P).mean()),
                variance_top=float((sv[:k] ** 2).sum() / (sv ** 2).sum()), top_k=k)


def cmd_entanglement(args) -> None:
    if os.path.exists(args.readout):
        print("[exchange rate] STS-B points lost per 0.01 of swap similarity removed\n")
        print(f"{'model':12s} {'edit':14s} {'rank':>5s} {'lambda':>7s} {'removed':>8s} "
              f"{'STS-B lost':>11s} {'rate':>6s}")
        rates = exchange_rates(args.readout)
        for model, rows in rates.items():
            for r in sorted(rows, key=lambda x: -x["removed"])[:6]:
                print(f"{model[:12]:12s} {r['edit']:14s} {r['rank']:5d} {r['lam']:7g} "
                      f"{r['removed']:8.4f} {r['stsb_lost']:11.4f} "
                      f"{r['stsb_lost'] / (r['removed'] / 0.01):6.3f}")
    else:
        rates = {}
        print(f"[exchange rate] skipped: {args.readout} not found")

    items = e2.role_items(args.per_pair, args.seed)
    result = {"exchange_rates": rates, "directions": {}}
    print(f"\n[directions] {len(items)} sentences; is there one role axis to amplify?\n")
    print(f"{'model':16s} {'names':>6s} {'role vs role':>13s} {'random':>8s} {'own name':>10s}"
          f" {'other names':>12s} {'top directions':>15s}")
    for name in args.models:
        m = chvlm.load(name, eager=False)
        E = chvlm.embed(m, [it["text"] for it in items], bs=128)
        R, I_names, _ = directions(E, items)
        ax = role_axis(R, I_names, args.rank)
        del m
        result["directions"][name] = ax
        print(f"{name:16s} {ax['n_names']:6d} {ax['shared_axis']:13.3f} {ax['random_level']:8.3f}"
              f" {ax['own_name']:10.3f} {ax['other_name']:12.3f} {ax['variance_top']:15.2f}")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "entanglement.json")
    json.dump(result, open(path, "w"), indent=2)
    print(f"\n[done] wrote {path}")
    print("reading: role-vs-role alignment at the random level means every name carries its own"
          " role direction, so there is no single direction for an edit to amplify; own name far"
          " above other names means the role rides on that name's identity direction")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("entanglement", help="price the trade-off and measure the overlap")
    e.add_argument("--models", nargs="+", default=["unsup-simcse", "retrained", "qwen3-0.6b"],
                   choices=list(chvlm.MODELS))
    e.add_argument("--readout", default="results/readout_edit.json")
    e.add_argument("--per-pair", type=int, default=2)
    e.add_argument("--rank", type=int, default=10, help="how many leading directions to report")
    e.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    {"entanglement": cmd_entanglement}[args.cmd](args)


if __name__ == "__main__":
    main()

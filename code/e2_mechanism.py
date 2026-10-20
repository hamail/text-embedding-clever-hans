"""
e2_mechanism.py -- what the similarity is made of.

Parts, run as subcommands:
  probe          Does the embedding record who did what to whom, or only word order?
                 (active and passive voice, a control task, calibration features)
  explain        Run BiLRP over a dataset and save one matrix per pair, with the
                 pair-level measures and the faithfulness test. Resumable: pairs whose
                 matrix already exists are skipped, so an interrupted run continues.

  measures       Turn saved matrices into the mechanism table: identical-word evidence,
                 the cross-position share, the per-cell ratio and the position-blind null,
                 by class. Works on runs from `explain` and on the old pipeline's runs,
                 recomputing every measure with the clean core's definitions, so a model
                 that need not be re-explained is still measured correctly.

  layers         Is it only the final layer? At every layer, how well a probe reads
                 roles, and whether that layer's similarity separates a role reversal
                 from a meaning-preserving rewrite. If some middle layer separates them,
                 the failure can be repaired after training by reading that layer.

  minimal-pairs  Does the similarity treat a role reversal like a harmless swap?
                 Two designs whose confounds run in opposite directions: verb contrast
                 (symmetric against asymmetric verbs, so the verb differs) and passive
                 contrast (same verb, but the meaning-keeping version adds "was" and
                 "by"). Equivalence tests on the similarities and, where explanations
                 exist, on the per-cell evidence.

Later parts will add: explain (BiLRP runs) and measures.

Usage (from ~/CHVLM/revised):
  python e2_mechanism.py probe
  python e2_mechanism.py probe --models unsup-simcse qwen3-0.6b
  python e2_mechanism.py minimal-pairs
  python e2_mechanism.py layers --models unsup-simcse qwen3-0.6b
  python e2_mechanism.py explain --model sup-simcse --dataset paws --n-per-class 400
  python e2_mechanism.py measures --dataset paws --legacy ../results/phase1_scale/paws
"""

from __future__ import annotations

import argparse
import json
import os
import random

import time

import numpy as np

import chvlm

OUT = "results/e2"

# =============================================================================
# Stimuli
# =============================================================================

NAMES = ("Smith Jones Taylor Wilson Davies Evans Johnson Roberts Walker Wright Robinson Thompson "
         "Hughes Edwards Harris Lewis Martin Jackson Clarke Turner Cooper Morris Watson Harrison "
         "Morgan Patel Allen Mitchell Anderson Phillips Parker Davis Garcia Miller Collins Stewart "
         "Murphy Campbell Kelly Cohen").split()
CITIES = "London Paris Berlin Madrid Rome Vienna Dublin Boston Chicago Toronto Sydney Tokyo".split()
FRAMES = [                       # no word here recurs within a sentence, in either voice
    "During {year}, {S} in {city}.",
    "{S} at the conference in {year}.",
    "The report says {S} in {city}.",
    "Last week, {S} in {city}.",
    "Earlier this year, {S} in private.",
]
VERBS = ("defeated hired sued trained thanked praised invited interviewed replaced criticised "
         "visited helped").split()     # asymmetric, past tense equal to past participle


PREDICATES = [                   # (keeps meaning when swapped, reverses roles), same length
    ("met", "defeated"), ("married", "hired"), ("divorced", "sued"), ("fought", "trained"),
    ("worked with", "worked for"), ("argued with", "reported to"),
    ("played against", "voted for"), ("competed against", "wrote to"),
]
MINIMAL_PAIRS = "data/minimal_pairs.jsonl"


def build_minimal_pairs(n_items: int = 400, seed: int = 7) -> list[dict]:
    """Matched twins: the same sentence with its two names swapped, once with a
    symmetric predicate (meaning kept) and once with an asymmetric one (roles reversed).
    The random draws follow the legacy script exactly, so the file is identical."""
    rng, seen, rows = random.Random(seed), set(), []
    while len(rows) < 2 * n_items:
        f, (sym, asym) = rng.choice(FRAMES), rng.choice(PREDICATES)
        a, b = rng.sample(NAMES, 2)
        kw = dict(year=rng.randint(1995, 2020), city=rng.choice(CITIES))
        key = (f, sym, a, b, kw["year"], kw["city"])
        if key in seen:
            continue
        seen.add(key); mid = len(rows) // 2
        for p, lab in ((sym, 1), (asym, 0)):
            rows.append(dict(s1=f.format(S=f"{a} {p} {b}", **kw), s2=f.format(S=f"{b} {p} {a}", **kw),
                             label=lab, match_id=mid, predicate=p))
    return rows


def role_items(per_pair: int, seed: int) -> list[dict]:
    """For every unordered name pair and each of `per_pair` samples (frame, verb, city,
    year), four sentences: each person as agent, in active and in passive voice."""
    rng = random.Random(seed)
    items = []
    for i, a in enumerate(NAMES):
        for b in NAMES[i + 1:]:
            for k in range(per_pair):
                f, v = rng.choice(FRAMES), rng.choice(VERBS)
                kw = dict(year=rng.randint(1995, 2020), city=rng.choice(CITIES))
                for agent, patient in ((a, b), (b, a)):
                    for voice, clause, first in (("active", f"{agent} {v} {patient}", agent),
                                                 ("passive", f"{patient} was {v} by {agent}", patient)):
                        items.append(dict(text=f.format(S=clause, **kw), agent=agent, patient=patient,
                                          first=first, voice=voice, pair=f"{a}|{b}", sample=k))
    return items


def splits(items: list[dict], seed: int):
    """Train; test A = name pairs never seen in training (generalisation);
    test B = seen pairs, held-out sample (for the control task)."""
    pairs = sorted({it["pair"] for it in items})
    rng = np.random.default_rng(seed); rng.shuffle(pairs)
    held = set(pairs[: len(pairs) // 5])
    last = max(it["sample"] for it in items)
    if last == 0:                                   # one sample per pair: no held-out sample exists
        train = [i for i, it in enumerate(items) if it["pair"] not in held]
        return train, [i for i, it in enumerate(items) if it["pair"] in held], []
    train = [i for i, it in enumerate(items) if it["pair"] not in held and it["sample"] != last]
    test_a = [i for i, it in enumerate(items) if it["pair"] in held]
    test_b = [i for i, it in enumerate(items) if it["pair"] not in held and it["sample"] == last]
    return train, test_a, test_b

# =============================================================================
# Probe
# =============================================================================

NAME_ID = {n: i for i, n in enumerate(NAMES)}


def fit(X: np.ndarray, y: np.ndarray, rows: list[int]):
    """A linear probe: standardised features, multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(X[rows], y[rows])


def two_name_score(pred: np.ndarray, target: np.ndarray, other: np.ndarray, rows: list[int]) -> dict:
    """Among predictions naming one of the two people, the share naming the target."""
    from scipy.stats import binomtest
    t, o = target[rows], other[rows]
    hit, other = pred == t, pred == o
    k, n = int(hit.sum()), int((hit | other).sum())
    lo, hi = chvlm.ci(chvlm.bootstrap(lambda ix: hit[ix].sum() / max((hit | other)[ix].sum(), 1), len(rows)))
    return dict(accuracy=k / n if n else float("nan"), ci=[lo, hi], n=n,
                named_one_of_two=float((hit | other).mean()),
                p_vs_half=float(binomtest(k, n, 0.5).pvalue) if n else float("nan"))


def role_score(pred: np.ndarray, items: list[dict], rows: list[int]) -> dict:
    """Role accuracy: 0.5 is chance; for passives probed after training on actives,
    below 0.5 means the probe follows word order rather than roles."""
    agent = np.array([NAME_ID[it["agent"]] for it in items])
    patient = np.array([NAME_ID[it["patient"]] for it in items])
    return two_name_score(pred, agent, patient, rows)


def evaluate(X: np.ndarray, items: list[dict], seed: int) -> dict:
    """Every probe condition for one set of features."""
    train, test_a, test_b = splits(items, seed)
    y = np.array([NAME_ID[it["agent"]] for it in items])
    voice = np.array([it["voice"] for it in items])
    act = [r for r in train if voice[r] == "active"]
    out = {}
    probe_act, probe_both = fit(X, y, act), fit(X, y, train)
    for name, probe in (("trained on active", probe_act), ("trained on both voices", probe_both)):
        for v in ("active", "passive"):
            rows = [r for r in test_a if voice[r] == v]
            out[f"{name} -> {v}"] = role_score(probe.predict(X[rows]), items, rows)
    # control task (after Hewitt and Liang 2019): each pair gets one random, fixed name as
    # its label, with no role content. A probe can only learn it by memorising pairs, so it
    # should succeed on held-out sentences of familiar pairs and fall to chance on unseen
    # pairs; that shows memorisation cannot explain the unseen-pair results above.
    rng = np.random.default_rng(seed + 1)
    pick = {p: int(rng.integers(2)) for p in sorted({it["pair"] for it in items})}
    ctrl = np.array([NAME_ID[it["pair"].split("|")[pick[it["pair"]]]] for it in items])
    ctrl_other = np.array([NAME_ID[it["pair"].split("|")[1 - pick[it["pair"]]]] for it in items])
    probe_ctrl = fit(X, ctrl, train)
    if test_b:                                      # needs at least two samples per pair
        out["control task -> familiar pairs"] = two_name_score(probe_ctrl.predict(X[test_b]), ctrl, ctrl_other, test_b)
    out["control task -> unseen pairs"] = two_name_score(probe_ctrl.predict(X[test_a]), ctrl, ctrl_other, test_a)
    return out


def calibration(items: list[dict]) -> dict[str, np.ndarray]:
    """Hand-built features that fix the scale of every condition."""
    n = len(NAMES)
    feats = {k: np.zeros((len(items), 2 * n)) for k in ("bag of names", "names by order", "names by role")}
    for r, it in enumerate(items):
        a, p, f = NAME_ID[it["agent"]], NAME_ID[it["patient"]], NAME_ID[it["first"]]
        second = p if f == a else a
        feats["bag of names"][r, [a, p]] = 1
        feats["names by order"][r, [f, n + second]] = 1
        feats["names by role"][r, [a, n + p]] = 1
    return {f"calibration: {k}": v for k, v in feats.items()}


def cmd_probe(args) -> None:
    items = role_items(args.per_pair, args.seed)
    train, test_a, test_b = splits(items, args.seed)
    print(f"[probe] {len(items)} sentences; train {len(train)}, unseen pairs {len(test_a)}, "
          f"seen pairs held-out samples {len(test_b)}")
    feats = calibration(items)
    for name in args.models:
        print(f"[encode] {name}", flush=True)
        m = chvlm.load(name, eager=False)
        feats[name] = chvlm.embed(m, [it["text"] for it in items], bs=128)
        del m
    result = {k: evaluate(X, items, args.seed) for k, X in feats.items()}

    cols = ["trained on active -> active", "trained on active -> passive",
            "trained on both voices -> active", "trained on both voices -> passive",
            "control task -> familiar pairs", "control task -> unseen pairs"]
    heads = ["act->act", "act->pas", "both->act", "both->pas", "ctrl-seen", "ctrl-new"]
    print(f"\n{'features':32s} " + " ".join(f"{h:>10s}" for h in heads))
    for k, r in result.items():
        print(f"{k:32s} " + " ".join(f"{r[c]['accuracy']:10.1%}" for c in cols))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"probe{args.tag}.json")
    json.dump(dict(args=vars(args), n_items=len(items), results=result), open(path, "w"), indent=2)
    print(f"\n[done] wrote {path}")
    print("reading: act->pas far below 50% = the probe follows word order, not roles; both->act and"
          " both->pas both above 50% = names are bound to roles in the embedding (an order code alone"
          " cannot exceed 50% on both voices); ctrl-new near 50% = memorisation cannot explain these")

# =============================================================================
# Minimal pairs: equivalence tests
# =============================================================================

MARGIN = 1.25                    # equivalence bounds 0.80-1.25, the bioequivalence standard, fixed in advance


def tost_log_ratio(r: np.ndarray, margin: float = MARGIN) -> dict:
    """Paired tests on per-item log ratios (reversed over kept), with a 90% interval.
      p_not_further  one-sided, H0: ratio >= margin. Rejecting it means a role reversal
                     moves the embedding no more than `margin` times as far as a harmless
                     swap: the role-blindness claim, since a role-aware model moves it further.
      p_tost         two-sided equivalence, ratio within [1/margin, margin]: "treated alike"."""
    from scipy.stats import t as tdist
    r = np.asarray(r)[np.isfinite(r)]
    n, mean = len(r), float(r.mean()) if len(r) else float("nan")
    se = float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0
    if not np.isfinite(mean) or se == 0:                      # a constant representation:
        return dict(n=n, ratio=float("nan"), ci90=[float("nan")] * 2,   # both distances are zero,
                    p_not_further=float("nan"), role_blind=False,       # so the ratio means nothing
                    p_tost=float("nan"), treated_alike=False, degenerate=True)
    L = np.log(margin)
    p_low = float(tdist.sf((mean + L) / se, n - 1))           # H0: ratio <= 1/margin
    p_high = float(tdist.cdf((mean - L) / se, n - 1))         # H0: ratio >= margin
    q = tdist.ppf(0.95, n - 1)
    return dict(n=n, ratio=float(np.exp(mean)), ci90=[float(np.exp(mean - q * se)), float(np.exp(mean + q * se))],
                p_not_further=p_high, role_blind=p_high < 0.05,
                p_tost=max(p_low, p_high), treated_alike=max(p_low, p_high) < 0.05)


def frame_percell(Bw: np.ndarray, wa: list[str], wb: list[str], predicate: str):
    """Per-cell ratio of the swapped names' evidence to that of the frame words: the
    predicate, the one thing that differs between twins, is left out of the reference."""
    A, pred = np.abs(Bw), {w.lower() for w in predicate.split()}
    cells = chvlm.identical_cells(wa, wb)
    cr = [A[i, j] for i, j, _, c in cells if c]
    al = [A[i, j] for i, j, _, c in cells if not c and wa[i] not in pred]
    return float(np.mean(cr) / np.mean(al)) if cr and al and np.mean(al) > 0 else None


def passive_items(n_items: int, seed: int) -> list[dict]:
    """Same verb throughout: a sentence, its passive (meaning kept, names reordered)
    and its role reversal (names reordered, meaning changed)."""
    rng, out = random.Random(seed), []
    for _ in range(n_items):
        f, v = rng.choice(FRAMES), rng.choice(VERBS)
        a, b = rng.sample(NAMES, 2)
        kw = dict(year=rng.randint(1995, 2020), city=rng.choice(CITIES))
        out.append(dict(base=f.format(S=f"{a} {v} {b}", **kw),
                        kept=f.format(S=f"{b} was {v} by {a}", **kw),
                        reversed=f.format(S=f"{b} {v} {a}", **kw)))
    return out


def cmd_minimal_pairs(args) -> None:
    rows = build_minimal_pairs()
    os.makedirs(os.path.dirname(MINIMAL_PAIRS), exist_ok=True)
    with open(MINIMAL_PAIRS, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    twins = {}
    for k, r in enumerate(rows):
        twins.setdefault(r["match_id"], {})[r["label"]] = k
    kept = [t[1] for t in twins.values()]; rev = [t[0] for t in twins.values()]
    result = {}
    print(f"[minimal-pairs] {len(twins)} matched twins; equivalence bounds {1 / MARGIN:.2f}-{MARGIN:.2f}")
    print(f"\n{'model':16s} {'kept cos':>9s} {'reversed':>9s} {'distance ratio (90% CI)':>26s}"
          f" {'P not further':>14s} {'P alike':>9s}  role-blind")
    for name in args.models:
        m = chvlm.load(chvlm.at_precision(name, args.precision), eager=False)
        E1 = chvlm.embed(m, [r["s1"] for r in rows], bs=128)
        E2 = chvlm.embed(m, [r["s2"] for r in rows], bs=128)
        del m
        mu = np.vstack([E1, E2]).mean(0)
        c = chvlm.cosine(E1, E2, mu)                               # centred on these sentences
        d = 1 - c
        res = tost_log_ratio(np.log(d[rev] / d[kept]))
        res.update(cos_kept=float(c[kept].mean()), cos_reversed=float(c[rev].mean()),
                   precision=chvlm.at_precision(name, args.precision).precision)
        result[name] = {"similarity": res}
        print(f"{name:16s} {res['cos_kept']:9.4f} {res['cos_reversed']:9.4f} "
              f"{res['ratio']:9.3f} ({res['ci90'][0]:.3f}-{res['ci90'][1]:.3f}) {res['p_not_further']:14.1e} "
              f"{res['p_tost']:9.1e}  {'yes' if res['role_blind'] else 'NO'}")
    for name, run in (args.mats or {}).items():                    # per-cell evidence, if explained
        recs = {(r["s1"], r["s2"]): r for r in map(json.loads, open(os.path.join(run, "records.jsonl")))}
        ratio = {}
        for k, r in enumerate(rows):
            rec = recs.get((r["s1"], r["s2"]))
            if rec is None:
                continue
            z = np.load(os.path.join(run, "mats", rec["pair"] + ".npz"), allow_pickle=True)
            wa = [chvlm.SPECIAL if w in ("[CLS]", "[SEP]", "[PAD]", "<SPECIAL>") else str(w) for w in z["words_a"]]
            wb = [chvlm.SPECIAL if w in ("[CLS]", "[SEP]", "[PAD]", "<SPECIAL>") else str(w) for w in z["words_b"]]
            ratio[k] = frame_percell(z["Bw"], wa, wb, r["predicate"])
        ok = [(ratio.get(a), ratio.get(b)) for a, b in zip(rev, kept)]
        ok = [(a, b) for a, b in ok if a and b]
        res = tost_log_ratio(np.log(np.array([a for a, _ in ok]) / np.array([b for _, b in ok])))
        result.setdefault(name, {})["per-cell evidence"] = res
        print(f"{name:16s} per-cell evidence, reversed/kept {res['ratio']:.3f} "
              f"({res['ci90'][0]:.3f}-{res['ci90'][1]:.3f})  P not further {res['p_not_further']:.1e}, "
              f"P alike {res['p_tost']:.1e}")
    # second design: same verb, the meaning-keeping version being the passive
    pit = passive_items(args.n_passive, args.seed)
    print(f"\n[passive contrast] {len(pit)} items, same verb in both conditions")
    print(f"{'model':16s} {'d kept':>8s} {'d reversed':>11s} {'distance ratio (90% CI)':>26s}"
          f" {'P not further':>14s} {'P alike':>9s}  role-blind")
    for name in args.models:
        m = chvlm.load(chvlm.at_precision(name, args.precision), eager=False)
        E = {k: chvlm.embed(m, [it[k] for it in pit], bs=128) for k in ("base", "kept", "reversed")}
        del m
        mu = np.vstack(list(E.values())).mean(0)
        dk = 1 - chvlm.cosine(E["base"], E["kept"], mu)
        dr = 1 - chvlm.cosine(E["base"], E["reversed"], mu)
        res = tost_log_ratio(np.log(dr / dk))
        res.update(d_kept=float(dk.mean()), d_reversed=float(dr.mean()))
        result.setdefault(name, {})["passive contrast"] = res
        print(f"{name:16s} {res['d_kept']:8.4f} {res['d_reversed']:11.4f} "
              f"{res['ratio']:9.3f} ({res['ci90'][0]:.3f}-{res['ci90'][1]:.3f}) {res['p_not_further']:14.1e} "
              f"{res['p_tost']:9.1e}  {'yes' if res['role_blind'] else 'NO'}")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"minimal_pairs{args.tag}.json")
    json.dump(dict(margin=MARGIN, results=result), open(path, "w"), indent=2)
    print(f"\n[done] wrote {path}")
    print("reading: role-blind 'yes' = a role reversal moves the embedding less than 1.25 times as far as a"
          " harmless swap; 'P alike' tests the stricter claim that the two are treated the same (0.80-1.25)."
          " The two designs have opposite biases: the verb contrast changes the verb, the passive contrast"
          " adds two words to the meaning-keeping version. A conclusion holding in both is safe.")

# =============================================================================
# Explanations
# =============================================================================

def cmd_explain(args) -> None:
    """BiLRP over one dataset, one matrix per pair."""
    pairs = (build_minimal_pairs() if args.dataset == "swapmin"
             else chvlm.load_pairs(args.dataset, args.n_per_class))
    out_dir = os.path.join(OUT, "explain", args.dataset, args.model)
    os.makedirs(os.path.join(out_dir, "mats"), exist_ok=True)
    m = chvlm.load(args.model)
    mu_path = os.path.join(out_dir, "mu.npy")
    if os.path.exists(mu_path):
        mu = np.load(mu_path)
    else:                                            # centring mean: the sentences of 128 pairs
        texts = [s for p in pairs[:128] for s in (p["s1"], p["s2"])]
        mu = chvlm.embed(m, texts).mean(0)
        np.save(mu_path, mu)
    rec_path = os.path.join(out_dir, "records.jsonl")
    done = {json.loads(l)["pair"] for l in open(rec_path)} if os.path.exists(rec_path) else set()
    print(f"[explain] {args.model} on {args.dataset}: {len(pairs)} pairs, {len(done)} already done")
    t0 = time.time()
    with chvlm.propagation_rules(m), open(rec_path, "a") as fh:
        for k, p in enumerate(pairs):
            pid = f"pair{k:04d}"
            if pid in done:
                continue
            e = chvlm.explain_pair(m, p["s1"], p["s2"])
            np.savez_compressed(os.path.join(out_dir, "mats", pid + ".npz"), Bw=e["Bw"],
                                words_a=np.array(e["words_a"]), words_b=np.array(e["words_b"]))
            psi_b = e["phi_b"] - mu
            fb, fr = chvlm.faithfulness(m, p["s1"], psi_b, mu, e["Bw"], e["words_a"])
            rec = dict(pair=pid, s1=p["s1"], s2=p["s2"],
                       **{k2: v for k2, v in p.items() if k2 in ("label", "score", "predicate", "match_id")},
                       cos_centred=float(np.dot(e["phi_a"] - mu, psi_b) /
                                         (np.linalg.norm(e["phi_a"] - mu) * np.linalg.norm(psi_b))),
                       faith_bilrp=fb, faith_random=fr,
                       **chvlm.pair_measures(e["Bw"], e["words_a"], e["words_b"]))
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (k + 1) % 50 == 0:
                rate = (time.time() - t0) / max(k + 1 - len(done), 1)
                print(f"    {k + 1}/{len(pairs)}  {rate:.1f}s/pair  "
                      f"{(len(pairs) - k - 1) * rate / 60:.0f} min left", flush=True)
    recs = [json.loads(l) for l in open(rec_path)]
    win = np.mean([r["faith_bilrp"] > r["faith_random"] for r in recs if r["faith_bilrp"] is not None])
    print(f"[done] {len(recs)} pairs in {out_dir}; faithfulness: BiLRP beats random on {win:.0%}")

# =============================================================================
# Measures
# =============================================================================

OLD_SPECIALS = {"[CLS]", "[SEP]", "[PAD]", "<SPECIAL>"}


def read_words(z):
    """Words of a saved matrix, with the old pipeline's special tokens mapped onto the
    clean core's SPECIAL, and E5's instruction prefix excluded as the core now does."""
    out = [chvlm.SPECIAL if str(w) in OLD_SPECIALS else str(w) for w in z]
    body = [i for i, w in enumerate(out) if w != chvlm.SPECIAL]
    if len(body) >= 2 and [out[i] for i in body[:2]] == ["query", ":"]:
        out[body[0]] = out[body[1]] = chvlm.SPECIAL
    return out


def run_measures(run: str) -> dict:
    """Every pair-level measure for one run directory, recomputed from its matrices."""
    recs = [json.loads(l) for l in open(os.path.join(run, "records.jsonl"))]
    rows = []
    for r in recs:
        p = os.path.join(run, "mats", r["pair"] + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        m = chvlm.pair_measures(z["Bw"], read_words(z["words_a"]), read_words(z["words_b"]))
        m["label"] = r.get("label")
        m["score"] = r.get("score")                          # graded datasets have no classes
        m["cos"] = r.get("cos_centred", r.get("cos_cen"))
        m["faith"] = (r.get("faith_bilrp") or 0) > (r.get("faith_random", r.get("faith_rand")) or 0)
        rows.append(m)
    return rows


def by_class(rows, key, stat=np.mean):
    """A statistic of one measure for each class, and the two-sided Mann-Whitney P."""
    from scipy.stats import mannwhitneyu
    a = [r[key] for r in rows if r["label"] == 1 and r[key] is not None]
    b = [r[key] for r in rows if r["label"] == 0 and r[key] is not None]
    if len(a) < 5 or len(b) < 5:
        return None, None, None
    return float(stat(a)), float(stat(b)), float(mannwhitneyu(a, b).pvalue)


def cmd_measures(args) -> None:
    runs = {}
    new_dir = os.path.join(OUT, "explain", args.dataset)
    # The legacy pipeline named a run directory after the checkpoint basename and this one
    # after the short name, so setdefault on the raw name lets the same model through
    # twice and "a new run wins" silently stops holding.  Key on the short name instead.
    short = {os.path.basename(s.hub).lower(): k for k, s in chvlm.MODELS.items()}
    key = lambda n: short.get(n.lower(), n)
    for d in (new_dir, args.legacy):
        if d and os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if os.path.exists(os.path.join(d, name, "records.jsonl")):
                    runs.setdefault(key(name), os.path.join(d, name))  # a new run wins over a legacy one
    if not runs:
        raise SystemExit(f"no runs found for {args.dataset}")
    result = {}
    graded = args.dataset == "stsb"
    print(f"[measures] {args.dataset}: {len(runs)} runs\n")
    if graded:                                               # no classes: report overall values
        print(f"{'model':26s} {'n':>5s} {'cosine':>8s} {'cross-position':>15s} {'per-cell':>9s}"
              f" {'overlap mass':>13s} {'mass vs score':>14s} {'faith':>6s}")
    else:
        print(f"{'model':26s} {'n 1/0':>9s} {'cosine 1/0':>17s} {'cross-position 1/0':>21s}"
              f" {'per-cell 1/0':>15s} {'null dev 1/0':>15s} {'faith':>6s}")
    for name, run in runs.items():
        rows = run_measures(run)
        faith_all = float(np.mean([r["faith"] for r in rows]))
        if graded:                                           # overall values, and how the
            from scipy.stats import spearmanr                # evidence tracks the human score
            g = np.array([r["score"] for r in rows if r["score"] is not None])
            mass = np.array([r["overlap_mass"] for r in rows if r["score"] is not None])
            pc = [r["percell_ratio"] for r in rows if r["percell_ratio"] is not None]
            rho = float(spearmanr(mass, g)[0]) if len(g) > 5 else float("nan")
            result[name] = dict(run=run, n=len(rows), cosine=float(np.mean([r["cos"] for r in rows])),
                                crosspos=float(np.mean([r["crosspos_share"] for r in rows])),
                                percell=float(np.median(pc)) if pc else None,
                                overlap_mass=float(mass.mean()), mass_vs_score=rho,
                                faithfulness=faith_all)
            r_ = result[name]
            print(f"{name[:26]:26s} {r_['n']:5d} {r_['cosine']:8.3f} {r_['crosspos']:15.3f} "
                  f"{(r_['percell'] or float('nan')):9.2f} {r_['overlap_mass']:13.3f} "
                  f"{r_['mass_vs_score']:14.3f} {faith_all:6.0%}")
            continue
        n1 = sum(r["label"] == 1 for r in rows); n0 = sum(r["label"] == 0 for r in rows)
        cos = by_class(rows, "cos"); cps = by_class(rows, "crosspos_share")
        pc = by_class(rows, "percell_ratio", np.median); nd = by_class(rows, "null_deviation")
        om = by_class(rows, "overlap_mass")
        faith = faith_all
        result[name] = dict(run=run, n=[n1, n0], cosine=cos, crosspos=cps, percell=pc,
                            null_deviation=nd, overlap_mass=om, faithfulness=faith)
        f = lambda t, d=3: "   n/a" if t[0] is None else f"{t[0]:.{d}f}/{t[1]:.{d}f}"
        p = lambda t: "" if t[0] is None else f" P={t[2]:.0e}"
        print(f"{name[:26]:26s} {n1:4d}/{n0:4d} {f(cos):>17s} {f(cps) + p(cps):>21s} "
              f"{f(pc, 2):>15s} {f(nd) :>15s} {faith:6.0%}")
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"measures_{args.dataset}.json")
    json.dump(result, open(path, "w"), indent=2)
    print(f"\n[done] wrote {path}")
    if graded:
        print("STS-B has graded scores, not classes: values are over all pairs, and the last"
              " column is how far identical-word evidence tracks the human score")
    else:
        print("class 1 = paraphrase or meaning kept, class 0 = non-paraphrase or roles reversed;"
              " per-cell is the median ratio of crossed to aligned identical-word evidence")

# =============================================================================
# Layers
# =============================================================================

def cmd_layers(args) -> None:
    """Role decodability and role separation, layer by layer."""
    items = role_items(args.per_pair, args.seed)
    train, test_a, _ = splits(items, args.seed)
    y = np.array([NAME_ID[it["agent"]] for it in items])
    voice = np.array([it["voice"] for it in items])
    pit = passive_items(args.n_passive, args.seed)
    result = {}
    for name in args.models:
        m = chvlm.load(name, eager=False)
        P = chvlm.embed_layers(m, [it["text"] for it in items], bs=128)
        S = {k: chvlm.embed_layers(m, [it[k] for it in pit], bs=128)
             for k in ("base", "kept", "reversed")}
        del m
        rows = []
        print(f"\n=== {name}: {P.shape[0] - 1} layers, {P.shape[1]} probe sentences")
        print(f"{'layer':>6s} {'probe act':>10s} {'probe pas':>10s} {'distance ratio':>15s}"
              f" {'90% CI':>17s}  separates reversals")
        for L in range(P.shape[0]):
            probe = fit(P[L], y, train)
            acc = {v: role_score(probe.predict(P[L][[r for r in test_a if voice[r] == v]]), items,
                                 [r for r in test_a if voice[r] == v])["accuracy"] for v in ("active", "passive")}
            mu = np.vstack([S[k][L] for k in S]).mean(0)
            dk = 1 - chvlm.cosine(S["base"][L], S["kept"][L], mu)
            dr = 1 - chvlm.cosine(S["base"][L], S["reversed"][L], mu)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = tost_log_ratio(np.log(dr / dk))
            rows.append(dict(layer=L, probe_active=acc["active"], probe_passive=acc["passive"], **t))
            if t.get("degenerate"):
                print(f"{L:6d} {acc['active']:10.1%} {acc['passive']:10.1%} "
                      f"{'constant representation, no distance to compare':>50s}")
            else:
                print(f"{L:6d} {acc['active']:10.1%} {acc['passive']:10.1%} {t['ratio']:15.3f} "
                      f"({t['ci90'][0]:.3f}-{t['ci90'][1]:.3f})  {'yes' if t['ci90'][0] > 1 else 'no'}")
        result[name] = rows
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "layers.json")
    json.dump(dict(args=vars(args), results=result), open(path, "w"), indent=2)
    print(f"\n[done] wrote {path}")
    print("reading: a distance ratio above 1 at some layer means that layer's similarity does separate"
          " role reversals, so reading it would repair the failure without retraining; ratios at or"
          " below 1 everywhere mean no layer does")

# =============================================================================
# Command line
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="does the embedding record roles or only word order?")
    p.add_argument("--models", nargs="+", default=["unsup-simcse", "sup-simcse", "bge-base",
                                                   "e5-base", "retrained", "qwen3-0.6b"],
                   choices=list(chvlm.MODELS))
    p.add_argument("--per-pair", type=int, default=4)
    p.add_argument("--tag", default="", help="suffix for the output file, to keep runs apart")
    p.add_argument("--seed", type=int, default=11)
    q = sub.add_parser("minimal-pairs", help="does the similarity treat role reversals like harmless swaps?")
    q.add_argument("--models", nargs="+", default=["unsup-simcse", "sup-simcse", "bge-base",
                                                   "e5-base", "retrained", "qwen3-0.6b"],
                   choices=list(chvlm.MODELS))
    q.add_argument("--n-passive", type=int, default=400)
    q.add_argument("--tag", default="", help="suffix for the output file, to keep runs apart")
    q.add_argument("--seed", type=int, default=11)
    q.add_argument("--precision", choices=["fp32", "fp16", "int8"], default=None,
                   help="override the registry, so sizes can be compared at one precision")
    q.add_argument("--mats", nargs="*", default=[], metavar="MODEL=RUN_DIR",
                   help="explanation runs for the per-cell test, e.g. "
                        "unsup-simcse=results/phase1_scale/swapmin/unsup-simcse-bert-base-uncased")
    x = sub.add_parser("explain", help="run BiLRP over a dataset")
    x.add_argument("--model", required=True, choices=list(chvlm.MODELS))
    x.add_argument("--dataset", required=True, choices=["paws", "qqp", "mrpc", "stsb", "swapmin"])
    x.add_argument("--n-per-class", type=int, default=400)
    mm = sub.add_parser("measures", help="the mechanism table from saved matrices")
    mm.add_argument("--dataset", required=True, choices=["paws", "qqp", "mrpc", "stsb", "swapmin"])
    mm.add_argument("--legacy", default=None, help="a directory of old-pipeline runs to include")
    l = sub.add_parser("layers", help="is it only the final layer?")
    l.add_argument("--models", nargs="+", default=["unsup-simcse", "qwen3-0.6b"], choices=list(chvlm.MODELS))
    l.add_argument("--per-pair", type=int, default=2)
    l.add_argument("--n-passive", type=int, default=400)
    l.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    if args.cmd == "minimal-pairs":
        args.mats = dict(x.split("=", 1) for x in args.mats)
    {"probe": cmd_probe, "minimal-pairs": cmd_minimal_pairs, "layers": cmd_layers,
     "explain": cmd_explain, "measures": cmd_measures}[args.cmd](args)


if __name__ == "__main__":
    main()

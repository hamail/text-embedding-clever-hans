"""
chvlm.py -- shared code for the revised paper. No experiments live here.

Sections
  1. Models        registry, loading, tokenisation, plain embeddings
  2. Words         token-to-word grouping, one definition of special tokens
  3. Explanation   propagation rules (as a context manager), relevances, BiLRP
  4. Measures      identical-word cells, cross-position share, per-cell ratio, null
  5. Data          sentence-pair datasets and the frozen blind sets
  6. Statistics    bootstrap, AUC, McNemar, partial Spearman

Conventions
  * Every special token -- [CLS], [SEP], <|endoftext|>, and any instruction
    prefix such as E5's "query: " -- becomes the word SPECIAL and is excluded
    from every measure, for every model family.
  * BiLRP matrices are the same with or without centring: gradient x input does
    not change when a constant is subtracted. Centring affects similarity only.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAU = 0.15                      # displacement above which an identical-word cell is "crossed"
SPECIAL = "<special>"           # the single word used for every special or prefix token
SQRT2 = 1.4142135623730951

# =============================================================================
# 1. Models
# =============================================================================

@dataclass(frozen=True)
class ModelSpec:
    hub: str                    # Hugging Face id or local checkpoint path
    family: str                 # "encoder" (BERT-style) or "decoder" (Qwen3-style)
    pooling: str                # "cls", "pooler" (CLS through BERT's MLP), "mean" or "last"
    prefix: str = ""            # instruction prefix, excluded from all measures
    precision: str = "fp32"     # "fp32", "fp16" or "int8"


MODELS = {
    "unsup-simcse": ModelSpec("princeton-nlp/unsup-simcse-bert-base-uncased", "encoder", "cls"),
    "sup-simcse":   ModelSpec("princeton-nlp/sup-simcse-bert-base-uncased", "encoder", "pooler"),
    "bge-base":     ModelSpec("BAAI/bge-base-en-v1.5", "encoder", "cls"),
    "e5-base":      ModelSpec("intfloat/e5-base-v2", "encoder", "mean", prefix="query: "),
    "gte-base":     ModelSpec("thenlper/gte-base", "encoder", "mean"),
    "mpnet-base":   ModelSpec("sentence-transformers/all-mpnet-base-v2", "encoder", "mean"),
    "mxbai-large":  ModelSpec("mixedbread-ai/mxbai-embed-large-v1", "encoder", "cls"),
    "retrained":    ModelSpec("ckpt/arm_a_s42", "encoder", "cls"),
    "qwen3-0.6b":   ModelSpec("Qwen/Qwen3-Embedding-0.6B", "decoder", "last"),
    "qwen3-4b":     ModelSpec("Qwen/Qwen3-Embedding-4B", "decoder", "last", precision="fp16"),
    "qwen3-8b":     ModelSpec("Qwen/Qwen3-Embedding-8B", "decoder", "last", precision="int8"),
}

FINETUNED = {                   # condition -> checkpoint path pattern, seeds 42, 1, 2
    "entity-swap":     "ckpt/ft_entity_s{seed}/step_00200",
    "matched-control": "ckpt/ft_control_s{seed}/step_00200",
    "no-negative":     "ckpt/ft_none_s{seed}/step_00200",
    "graded":          "ckpt/arm_a_s42_stsb{suffix}",
}


def finetuned(condition: str, seed: int) -> ModelSpec:
    """The spec of one fine-tuned checkpoint of the retrained model."""
    if condition == "graded":
        path = FINETUNED["graded"].format(suffix="" if seed == 42 else f"_s{seed}")
    else:
        path = FINETUNED[condition].format(seed=seed)
    return ModelSpec(path, "encoder", "cls")


@dataclass
class Loaded:
    name: str
    spec: ModelSpec
    tok: object
    model: torch.nn.Module


def at_precision(name: str | ModelSpec, precision: str | None) -> ModelSpec:
    """The same model at a stated precision. Sizes in a family ship at different
    precisions by default, so a comparison across sizes must fix this or the trend is
    confounded: quantisation noise inflates every distance and drives ratios toward 1."""
    import dataclasses
    spec = MODELS[name] if isinstance(name, str) else name
    return spec if not precision else dataclasses.replace(spec, precision=precision)


def load(name: str | ModelSpec, eager: bool = True) -> Loaded:
    """Load a model frozen, in evaluation mode. Eager attention keeps the
    attention probabilities visible to the propagation rules."""
    from transformers import AutoModel, AutoTokenizer
    spec = MODELS[name] if isinstance(name, str) else name
    kw = {"attn_implementation": "eager"} if eager else {}
    import transformers
    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    dtype_arg = "dtype" if ver >= (4, 56) else "torch_dtype"      # renamed in 4.56
    if spec.precision == "int8":
        from transformers import BitsAndBytesConfig
        kw.update(quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
    else:
        kw[dtype_arg] = torch.float16 if spec.precision == "fp16" else torch.float32
    tok = AutoTokenizer.from_pretrained(spec.hub)
    model = AutoModel.from_pretrained(spec.hub, **kw)
    if spec.precision != "int8":
        model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Loaded(name if isinstance(name, str) else spec.hub, spec, tok, model)


def _ids(m: Loaded, text: str, max_length: int):
    """Token ids and character offsets, at most max_length tokens. A decoder's
    sequence ends in exactly one special token, the one it pools from.
    Qwen3-Embedding's tokenizer appends <|endoftext|> itself; an end token is
    added only when the tokenizer does not end the sequence with a special token."""
    enc = m.tok(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    ids, offsets = list(enc["input_ids"]), list(enc["offset_mapping"])
    if m.spec.family == "decoder" and ids[-1] not in set(m.tok.all_special_ids or []):
        ids = ids[:max_length - 1] + [m.tok.eos_token_id]
        offsets = offsets[:max_length - 1] + [(0, 0)]
    return ids, offsets


def tokenize(m: Loaded, text: str, max_length: int = 64):
    """Token ids, attention mask and a per-token prefix flag for one sentence."""
    ids, offsets = _ids(m, m.spec.prefix + text, max_length)
    n_pre = len(m.spec.prefix)
    in_prefix = [0 < e <= n_pre for _, e in offsets] if n_pre else [False] * len(ids)
    dev = next(m.model.parameters()).device
    ids_t = torch.tensor([ids], device=dev)
    return ids_t, torch.ones_like(ids_t), in_prefix


def _pool(h: torch.Tensor, amask: torch.Tensor, pooling: str, model=None,
          last_idx: int | None = None) -> torch.Tensor:
    """Sentence embeddings from hidden states [B, T, d].
    "last" reads the final unpadded position, or last_idx when given, so that
    masking words (faithfulness test) cannot move the pooled position.
    "pooler" applies BERT's MLP pooler, tanh(W h_cls + b), written as
    z * [tanh(z)/z] with the ratio detached: the forward value is unchanged,
    and the tanh is treated as locally constant like every other nonlinearity."""
    if pooling == "cls":
        return h[:, 0]
    if pooling == "pooler":
        z = model.pooler.dense(h[:, 0])
        return z * (torch.tanh(z) / (z + torch.where(z >= 0, 1e-9, -1e-9))).detach()
    if pooling == "last":
        idx = amask.sum(1) - 1 if last_idx is None else torch.full((h.shape[0],), last_idx, device=h.device)
        return h[torch.arange(h.shape[0], device=h.device), idx]
    m = amask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1)


@torch.no_grad()
def embed(m: Loaded, texts: list[str], bs: int = 64, max_length: int = 64) -> np.ndarray:
    """Plain sentence embeddings, float32, uncentred."""
    out = []
    dev = next(m.model.parameters()).device
    for s in range(0, len(texts), bs):
        batch = [m.spec.prefix + t for t in texts[s:s + bs]]
        if m.spec.family == "decoder":                        # same tokens as tokenize()
            ids = [_ids(m, t, max_length)[0] for t in batch]
            L = max(map(len, ids)); pad = m.tok.pad_token_id or m.tok.eos_token_id
            inp = torch.tensor([x + [pad] * (L - len(x)) for x in ids], device=dev)
            am = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids], device=dev)
        else:
            enc = m.tok(batch, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(dev)
            inp, am = enc["input_ids"], enc["attention_mask"]
        h = m.model(input_ids=inp, attention_mask=am).last_hidden_state
        out.append(_pool(h, am, m.spec.pooling, m.model).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def embed_layers(m: Loaded, texts: list[str], bs: int = 64, max_length: int = 64) -> np.ndarray:
    """Pooled representations at every layer: [n_layers + 1, n_texts, d], layer 0 being
    the embedding layer. Pooling follows the model's own readout, except that "pooler"
    falls back to CLS, since BERT's MLP pooler exists only on top of the last layer."""
    pooling = "cls" if m.spec.pooling == "pooler" else m.spec.pooling
    dev = next(m.model.parameters()).device
    out = []
    for s in range(0, len(texts), bs):
        batch = [m.spec.prefix + t for t in texts[s:s + bs]]
        if m.spec.family == "decoder":
            ids = [_ids(m, t, max_length)[0] for t in batch]
            L = max(map(len, ids)); pad = m.tok.pad_token_id or m.tok.eos_token_id
            inp = torch.tensor([x + [pad] * (L - len(x)) for x in ids], device=dev)
            am = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids], device=dev)
        else:
            enc = m.tok(batch, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(dev)
            inp, am = enc["input_ids"], enc["attention_mask"]
        hs = m.model(input_ids=inp, attention_mask=am, output_hidden_states=True).hidden_states
        out.append(torch.stack([_pool(h, am, pooling) for h in hs]).float().cpu())
    return torch.cat(out, dim=1).numpy()


def cosine(a: np.ndarray, b: np.ndarray, mu: np.ndarray | None = None) -> np.ndarray:
    """Row-wise cosine, optionally after subtracting a centring mean."""
    if mu is not None:
        a, b = a - mu, b - mu
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

# =============================================================================
# 2. Words
# =============================================================================

def _is_punct(piece: str) -> bool:
    """A token made only of punctuation or symbols."""
    return bool(piece) and not any(ch.isalnum() for ch in piece)


def words(m: Loaded, ids: torch.Tensor, in_prefix: list[bool]):
    """Group tokens into words. Returns (words, groups) with every special or
    prefix token mapped to SPECIAL; words are lower-cased. Punctuation is always
    its own word, in every family: BPE would otherwise glue "." onto the name
    before it, and "Mary." would not match "Mary"."""
    toks = m.tok.convert_ids_to_tokens(ids[0].tolist())
    special_ids = set(m.tok.all_special_ids or [])
    out_w, out_g = [], []
    for i, t in enumerate(toks):
        if ids[0, i].item() in special_ids or in_prefix[i]:
            out_w.append(SPECIAL); out_g.append([i]); continue
        if m.spec.family == "encoder":                       # WordPiece: "##" continues
            cont = t.startswith("##")
            piece = t[2:] if cont else t
        else:                                                # BPE: "Ġ" starts a word
            clean = t.replace("\u0120", " ").replace("\u010a", " ")
            cont = not clean.startswith(" ") and i > 0
            piece = clean.strip()
        cont = (cont and bool(out_w) and out_w[-1] != SPECIAL
                and not _is_punct(out_w[-1]) and not _is_punct(piece))
        if cont:
            out_w[-1] += piece.lower(); out_g[-1].append(i)
        else:
            out_w.append(piece.lower()); out_g.append([i])
    return out_w, out_g

# =============================================================================
# 3. Explanation
# =============================================================================

def _selftest(m: Loaded) -> None:
    """Fail loudly unless every propagation rule is active."""
    dev = next(m.model.parameters()).device
    x = torch.randn(2, 5, device=dev, requires_grad=True)
    assert not F.softmax(x, dim=-1).requires_grad, "softmax rule inactive"
    if m.spec.family == "encoder":
        norm = m.model.encoder.layer[0].output.LayerNorm
        mlp = m.model.encoder.layer[0].intermediate
        width = mlp.dense.in_features
        def true_mlp(z):
            h = mlp.dense(z); return 0.5 * h * (1.0 + torch.erf(h / SQRT2))
    else:
        norm = m.model.layers[0].input_layernorm
        mlp = m.model.layers[0].mlp
        width = mlp.gate_proj.in_features
        def true_mlp(z):
            return mlp.down_proj(_ORIG["silu"](mlp.gate_proj(z)) * mlp.up_proj(z))
    dt = next(m.model.parameters()).dtype
    z = torch.randn(1, 4, norm.weight.shape[0], device=dev, dtype=dt, requires_grad=True)
    g = torch.autograd.grad(norm(z).sum(), z)[0]
    assert (g * z).sum().abs().item() > 1e-4, "normalisation rule inactive"
    z = torch.randn(1, 3, width, device=dev, dtype=dt, requires_grad=True)
    gp = torch.autograd.grad(mlp(z).sum(), z)[0]
    z2 = z.detach().clone().requires_grad_(True)
    gt = torch.autograd.grad(true_mlp(z2).sum(), z2)[0]
    assert ((gp - gt).norm() / (gt.norm() + 1e-9)).item() > 1e-3, "MLP rule inactive"


_ORIG = {"softmax": F.softmax, "silu": F.silu}


@contextlib.contextmanager
def propagation_rules(m: Loaded):
    """Treat the transformer's nonlinear pathways as locally constant, so that
    gradient x input on the embedding layer gives the relevances. Encoders:
    attention, LayerNorm scale and GELU gate. Decoders: attention, RMSNorm scale
    and SwiGLU gate. Forward values are unchanged; everything is restored on exit."""
    saved = {"F.softmax": F.softmax, "nn.softmax": torch.nn.functional.softmax,
             "LayerNorm": torch.nn.LayerNorm.forward}

    def softmax_detached(input, dim=None, _stacklevel=3, dtype=None):
        return _ORIG["softmax"](input, dim=dim, dtype=dtype).detach()
    F.softmax = torch.nn.functional.softmax = softmax_detached

    if m.spec.family == "encoder":
        import transformers.models.bert.modeling_bert as mb
        saved["BertIntermediate"] = mb.BertIntermediate.forward

        def layernorm(self, x):
            mu = x.mean(-1, keepdim=True)
            std = torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + self.eps)
            return (x - mu) / std.detach() * self.weight + self.bias

        def intermediate(self, hidden_states):
            h = self.dense(hidden_states)
            g = 0.5 * h * (1.0 + torch.erf(h / SQRT2))
            return h * (g / (h + torch.where(h >= 0, 1e-9, -1e-9))).detach()
        torch.nn.LayerNorm.forward = layernorm
        mb.BertIntermediate.forward = intermediate
    else:
        import transformers.models.qwen3.modeling_qwen3 as mq
        saved["RMSNorm"], saved["MLP"] = mq.Qwen3RMSNorm.forward, mq.Qwen3MLP.forward

        def rmsnorm(self, x):
            eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps).detach() * self.weight

        def swiglu(self, x):
            return self.down_proj(_ORIG["silu"](self.gate_proj(x)).detach() * self.up_proj(x))
        mq.Qwen3RMSNorm.forward, mq.Qwen3MLP.forward = rmsnorm, swiglu
    try:
        _selftest(m)
        yield
    finally:
        F.softmax = saved["F.softmax"]
        torch.nn.functional.softmax = saved["nn.softmax"]
        torch.nn.LayerNorm.forward = saved["LayerNorm"]
        if m.spec.family == "encoder":
            import transformers.models.bert.modeling_bert as mb
            mb.BertIntermediate.forward = saved["BertIntermediate"]
        else:
            import transformers.models.qwen3.modeling_qwen3 as mq
            mq.Qwen3RMSNorm.forward, mq.Qwen3MLP.forward = saved["RMSNorm"], saved["MLP"]


def relevance(m: Loaded, ids: torch.Tensor, amask: torch.Tensor, chunk: int = 32):
    """Embedding phi [d] and per-token relevances R [T, d], R[t, k] being the
    gradient x input of phi_k at token t. Call inside propagation_rules(m)."""
    if m.spec.family == "encoder":
        with torch.no_grad():
            emb0 = m.model.embeddings(input_ids=ids)
        emb = emb0.detach().clone().requires_grad_(True)
        if hasattr(m.model, "get_extended_attention_mask"):          # transformers 4.x
            ext = m.model.get_extended_attention_mask(amask, ids.shape)
        else:                                                         # 5.x: the same additive mask
            ext = (1.0 - amask[:, None, None, :].to(emb.dtype)) * torch.finfo(emb.dtype).min
        h = m.model.encoder(emb, attention_mask=ext).last_hidden_state
    else:
        emb = m.model.embed_tokens(ids).detach().clone().requires_grad_(True)
        h = m.model(inputs_embeds=emb, attention_mask=amask).last_hidden_state
    phi = _pool(h, amask, m.spec.pooling, m.model)[0]
    d, T = phi.shape[0], ids.shape[1]
    R = torch.zeros(T, d, device=phi.device)
    eye = torch.eye(d, device=phi.device, dtype=phi.dtype)
    for s in range(0, d, chunk):
        try:
            g = torch.autograd.grad(phi, emb, grad_outputs=eye[s:s + chunk],
                                    retain_graph=True, is_grads_batched=True)[0]
            R[:, s:s + chunk] = (g[:, 0] * emb[0].unsqueeze(0)).sum(-1).T.float()
        except RuntimeError:                                 # memory: fall back to one at a time
            for k in range(s, min(s + chunk, d)):
                g = torch.autograd.grad(phi[k], emb, retain_graph=True)[0]
                R[:, k] = (g[0] * emb[0]).sum(-1).float()
    out = phi.detach().float().cpu(), R.detach().cpu()
    del h, phi, R, emb, eye
    gc.collect(); torch.cuda.empty_cache()
    return out


def aggregate(B: np.ndarray, ga: list[list[int]], gb: list[list[int]]) -> np.ndarray:
    """Sum token-pair relevances into word-pair relevances."""
    return np.array([[B[np.ix_(g1, g2)].sum() for g2 in gb] for g1 in ga])


def explain_pair(m: Loaded, s1: str, s2: str) -> dict:
    """BiLRP for one sentence pair. Call inside propagation_rules(m)."""
    ia, ma, pa = tokenize(m, s1)
    ib, mb_, pb = tokenize(m, s2)
    phi_a, Ra = relevance(m, ia, ma)
    phi_b, Rb = relevance(m, ib, mb_)
    wa, ga = words(m, ia, pa)
    wb, gb = words(m, ib, pb)
    Bw = aggregate((Ra @ Rb.T).numpy(), ga, gb)
    return dict(phi_a=phi_a.numpy(), phi_b=phi_b.numpy(), Bw=Bw, words_a=wa, words_b=wb)

# =============================================================================
# 4. Measures
# =============================================================================

def _positions(ws: list[str]) -> dict[int, float]:
    """Relative position in [0, 1] of each non-special word."""
    idx = [i for i, w in enumerate(ws) if w and w != SPECIAL]
    return {i: k / max(len(idx) - 1, 1) for k, i in enumerate(idx)}


def identical_cells(wa: list[str], wb: list[str], tau: float = TAU):
    """(i, j, displacement, crossed) for every pair of identical non-special words."""
    ra, rb = _positions(wa), _positions(wb)
    return [(i, j, abs(ra[i] - rb[j]), abs(ra[i] - rb[j]) > tau)
            for i in ra for j in rb if wa[i] == wb[j]]


def pair_measures(Bw: np.ndarray, wa: list[str], wb: list[str], tau: float = TAU) -> dict:
    """All pair-level measures of the paper from one word-pair matrix.
      overlap_mass     share of non-special evidence on identical-word cells
      crosspos_share   share of identical-word evidence on crossed cells (equation 2)
      special_share    share of all evidence on special tokens
      percell_ratio    mean crossed cell over mean aligned cell (None if either is absent)
      null_cross       share of identical-word cells that are crossed (position-blind null)
      null_deviation   crosspos_share minus null_cross"""
    A = np.abs(Bw)
    sa = np.array([w == SPECIAL or not w for w in wa]); sb = np.array([w == SPECIAL or not w for w in wb])
    special = sa[:, None] | sb[None, :]
    total = A.sum() + 1e-12
    content = A[~special].sum() + 1e-12
    cells = identical_cells(wa, wb, tau)
    cr = [A[i, j] for i, j, _, c in cells if c]
    al = [A[i, j] for i, j, _, c in cells if not c]
    ident = sum(cr) + sum(al)
    out = dict(overlap_mass=float(ident / content), special_share=float(A[special].sum() / total),
               crosspos_share=float(sum(cr) / (ident + 1e-12)), n_crossed=len(cr), n_aligned=len(al),
               percell_ratio=None, null_cross=None, null_deviation=None)
    if cr and al and sum(al) > 0:
        out["percell_ratio"] = float(np.mean(cr) / np.mean(al))
        out["null_cross"] = len(cr) / len(cells)
        out["null_deviation"] = out["crosspos_share"] - out["null_cross"]
    return out


def faithfulness(m: Loaded, s1: str, psi_b: np.ndarray, mu: np.ndarray, Bw: np.ndarray,
                 wa: list[str], k: int = 3, n_rand: int = 3, seed: int = 0):
    """Similarity drop from deleting the k most relevant words of sentence 1
    against the mean drop from deleting k random non-special words."""
    ids, am, pre = tokenize(m, s1)
    _, groups = words(m, ids, pre)
    cand = [i for i, w in enumerate(wa) if w != SPECIAL]
    if len(cand) <= k:
        return None, None

    @torch.no_grad()
    def score(mask):
        h = m.model(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = _pool(h, mask, m.spec.pooling, m.model, last_idx=ids.shape[1] - 1)
        return float((pooled[0].float().cpu().numpy() - mu) @ psi_b)

    def occlude(word_idx):
        mask = am.clone()
        for w in word_idx:
            mask[0, groups[w]] = 0
        return score(mask)
    base = score(am)
    rel = np.abs(Bw).sum(1)
    top = [i for i in np.argsort(-rel) if i in cand][:k]
    rng = np.random.default_rng(seed)
    rand = [base - occlude(list(rng.choice(cand, size=k, replace=False))) for _ in range(n_rand)]
    return base - occlude(top), float(np.mean(rand))

# =============================================================================
# 5. Data
# =============================================================================

STOPWORDS = set(("a an the of to in on for at by with from and or but is are was were be been "
                 "being it its this that these those as he she they we you i his her their our "
                 "your not no do does did has have had will would can could should may might").split())


def content_words(s: str) -> set[str]:
    """Lower-cased content words of a sentence."""
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    """Content-word Jaccard overlap of two sentences."""
    A, B = content_words(a), content_words(b)
    return len(A & B) / max(len(A | B), 1)


def load_pairs(dataset: str, n_per_class: int, min_jaccard: float = 0.4, seed: int = 123):
    """Development sentence pairs: paws, qqp, mrpc (binary, overlap-filtered),
    stsb (graded), or swapmin (the minimal pairs). Returns a list of dicts."""
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    if dataset == "stsb":
        ds = load_dataset("nyu-mll/glue", "stsb", split="validation")
        idx = rng.permutation(len(ds))[:n_per_class]
        return [dict(s1=ds[int(i)]["sentence1"], s2=ds[int(i)]["sentence2"],
                     score=float(ds[int(i)]["label"])) for i in idx]
    if dataset == "paws":
        ds = load_dataset("google-research-datasets/paws", "labeled_final", split="validation")
        rows = [(r["sentence1"], r["sentence2"], r["label"]) for r in ds]
    elif dataset == "qqp":
        rows = [(r["question1"], r["question2"], r["label"]) for r in load_dataset("nyu-mll/glue", "qqp", split="validation")]
    elif dataset == "mrpc":
        rows = [(r["sentence1"], r["sentence2"], r["label"]) for split in ("validation", "test")
                for r in load_dataset("nyu-mll/glue", "mrpc", split=split)]
    elif dataset == "swapmin":
        rows = [(r["s1"], r["s2"], r["label"]) for r in map(json.loads, open("data/minimal_pairs.jsonl"))]
    else:
        raise ValueError(dataset)
    out = []
    for lab in (1, 0):
        cand = [(a, b) for a, b, l in rows if l == lab and jaccard(a, b) >= min_jaccard]
        for i in rng.permutation(len(cand))[:n_per_class]:
            out.append(dict(s1=cand[i][0], s2=cand[i][1], label=lab))
    return out


def load_blind(name: str, protocol_commit: str = "01337a7") -> list[dict]:
    """A frozen blind set. Refuses to load a file whose checksum differs from the
    manifest committed with the protocol."""
    path = os.path.join("data/blind", f"{name}.jsonl")
    manifest = json.load(open("data/blind/MANIFEST.json"))
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != manifest[name]["sha256"]:
        raise RuntimeError(f"{path} differs from the manifest frozen at {protocol_commit}")
    return [json.loads(l) for l in open(path, encoding="utf-8")]

# =============================================================================
# 6. Statistics
# =============================================================================

def bootstrap(stat, n: int, reps: int = 2000, seed: int = 0) -> np.ndarray:
    """Values of stat(index_array) over bootstrap resamples of n items."""
    rng = np.random.default_rng(seed)
    return np.array([stat(rng.integers(0, n, n)) for _ in range(reps)])


def ci(values: np.ndarray, level: float = 0.90) -> tuple[float, float]:
    """Percentile interval of bootstrap values.

    The default is 0.90, the level the manuscript reports, so a caller that omits it
    gets the level the paper claims.  This does NOT touch the equivalence intervals in
    T01, T01b, T02 or T02b: tost_log_ratio computes those analytically from the paired
    log ratios, with tdist.ppf(0.95) on each side, which is already a 90% interval.
    What this default governs is the bootstrap intervals in the probe code and in the
    superposition decomposition, neither of which is printed as a table column, though
    the probe interval may reach Figure 3 as an error bar.
    """
    a = (1 - level) / 2
    return float(np.percentile(values, 100 * a)), float(np.percentile(values, 100 * (1 - a)))


def auc_ci(pos: np.ndarray, neg: np.ndarray, reps: int = 400, seed: int = 0,
           level: float = 0.90) -> tuple[float, float]:
    """Percentile interval for auc(pos, neg), each arm resampled at its OWN length.

    Resampling both arms at len(pos) + len(neg) and folding the draw with a modulo
    goes wrong twice.  The interval narrows by roughly the square root of the length
    ratio; and where the negative pool is several subcases laid end to end, the fold
    double-counts the leading block.  On HANS, with two non-entailment subcases of
    equal size, that put two thirds of the weight on the first of them, so the
    interval estimated a differently weighted AUC than the point estimate and came
    out entirely below it for every model.
    """
    rng = np.random.default_rng(seed)
    v = np.array([auc(pos[rng.integers(0, len(pos), len(pos))],
                      neg[rng.integers(0, len(neg), len(neg))]) for _ in range(reps)])
    return ci(v, level)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random positive outscores a random negative (ties count half)."""
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def mcnemar(before: np.ndarray, after: np.ndarray) -> tuple[int, int, float]:
    """Items fixed and broken moving from before to after (True = failure),
    with the exact McNemar P."""
    from scipy.stats import binomtest
    fixed, broke = int((before & ~after).sum()), int((~before & after).sum())
    p = binomtest(fixed, fixed + broke, 0.5).pvalue if fixed + broke else 1.0
    return fixed, broke, float(p)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman partial correlation of x and y, controlling for z."""
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    Z = np.c_[np.ones_like(rz), rz]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])

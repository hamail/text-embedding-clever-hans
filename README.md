# Who does what to whom: sentence embedding similarities ignore semantic roles

Code, stimuli and generated tables for the paper. Every number quoted in the
manuscript is a cell of one of the table bodies in `results/tables/`, and
`code/checknums.py` checks the paper against them.

```bash
python code/checknums.py
```

## Layout

| path | what it holds |
|---|---|
| `code/` | the analysis pipeline, the figure scripts and `checknums.py` |
| `code/e2_mechanism.py` | the stimulus generator: five frames, eight predicate pairs, twelve passive-contrast verbs, seeds |
| `data/minimal_pairs.jsonl` | the matched twins: 800 lines, 400 items paired by `match_id` |
| `data/passive_contrast.jsonl` | the 400 passive-contrast triples |
| `data/examples.json` | the ten worked items, one per domain |
| `data/audit_sample.tsv` | the 100-sentence news sample searched for naturally occurring reversals |
| `data/blind/MANIFEST.json` | which release of each public dataset was used |
| `results/tables/` | every generated table, as `.tex` and `.csv` |
| `results/` | the per-run outputs the tables are built from |

### The matched twins

Each line is one sentence pair. `match_id` joins the two pairs of an item.
`label` 1 marks the pair whose predicate is symmetric, where exchanging the two
names preserves the meaning, and 0 marks the pair whose predicate is asymmetric,
where the identical exchange reverses the roles. The ratio tested in the paper
is the distance across the second divided by the distance across the first.

## Scripts

| script | what it produces |
|---|---|
| `checknums.py` | verifies every numerical claim in the paper against the generated tables |
| `e2_mechanism.py` | the matched twins and the passive contrast, at fixed seeds |
| `tables.py` | table bodies T01 to T18 |
| `make_T19.py` | T19, the positive control, from `results/e7/positive_control.json` |
| `twin_item_rates.py` | item and entity-pair bootstrap intervals, and the per-predicate breakdown |
| `make_stab_predicates.py` | T23, the per-predicate table |
| `semeval_probe.py` | the base rate of each relation direction, and the sampled-text items |
| `semeval_ratio.py` | the directed against *Other* comparison on sampled text |

`checknums.py`, the stimulus generation and the statistical analyses run on CPU
in minutes. Reproducing the embeddings themselves needs a GPU, since the set
includes decoder models up to 8B parameters.

## Table numbering

The file names here are not the numbers in the paper, so the mapping is below.
Note in particular that `T01` is the passive contrast and `T02` is the matched
twins, the opposite order from the supplement.

| file | paper |
|---|---|
| `T02` | Supplementary Table 1, role blindness on the matched twins |
| `T01b` | Supplementary Table 2, precision crossed with size |
| `T02b` | Supplementary Table 3, per-cell evidence, predicate excluded |
| `T01` | Supplementary Table 4, the passive contrast |
| `T04` | Supplementary Table 5, the full layer sweep |
| `T05` | Supplementary Table 6, explanation measures by dataset |
| `T03` | Supplementary Table 7, the linear probe |
| `T09` | Supplementary Table 8, role directions and entity identity |
| `T12` | Supplementary Table 9, the binding term decomposed |
| `T13` | Supplementary Table 10, rank growth with the vocabulary |
| `T10` | Supplementary Table 11, what separating role swaps costs |
| `T06` | Supplementary Table 12, graded supervision and its control |
| `T08` | Supplementary Table 13, scores per checkpoint |
| `T11` | Supplementary Table 14, the zero-shot retrieval cost |
| `T14` | Supplementary Table 15, H1, HANS |
| `T15` | Supplementary Table 16, H2, PAWS-X |
| `T16` | Supplementary Table 17, H3, the benchmark excess |
| `T17` | Supplementary Table 18, H4, transfer |
| `data/examples.json` | Supplementary Table 19, the ten items as written |
| `T18` | Supplementary Table 20, distances for every item and model |
| `T19` | Supplementary Table 21, the positive control |
| from `twin_item_rates.py` | Supplementary Table 22, margin sensitivity |
| `T23` | Supplementary Table 23, the per-predicate breakdown |
| from `semeval_probe.py` | Supplementary Table 24, both directions in natural text |
| from `semeval_ratio.py` | Supplementary Table 25, the same measurement on sampled text |
| `T07` | Table 1 of the main text, what the negative has to be |

## Names used in the code and in the paper

The code and the generated files carry older names for three things. They are
the same objects throughout.

| code | paper |
|---|---|
| entity swap | role swap |
| control swap | non-role swap |
| verb contrast | matched twins |

## Public datasets

STS-B, SICK-R, STS 2012–2016, PAWS, PAWS-X, HANS, MRPC, NanoBEIR and
SemEval-2010 Task 8 are not redistributed here. `data/blind/MANIFEST.json`
records which release of each was used, and each is obtained from its own
distribution.

## Licence

MIT, see `LICENSE`.

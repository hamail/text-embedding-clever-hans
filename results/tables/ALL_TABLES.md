# Tables

Built by tables.py from 2026-09-28T15:50:23. Every number comes from a run's own JSON; none is typed in.

## T01. The reversal is closer than the paraphrase
*Source: e2 minimal-pairs*

| model | d to kept meaning | d to reversed | ratio | 90% CI | P not further |
|---|---|---|---|---|---|
| retrained | 0.0277 | 0.0045 | 0.158 | 0.153-0.163 | 4.6e-300 |
| unsup-SimCSE | 0.0492 | 0.0098 | 0.184 | 0.177-0.191 | 9.3e-251 |
| E5-base | 0.0275 | 0.0078 | 0.265 | 0.254-0.276 | 4.3e-210 |
| sup-SimCSE | 0.0241 | 0.0078 | 0.312 | 0.303-0.321 | 2.7e-246 |
| BGE-base | 0.0270 | 0.0103 | 0.346 | 0.331-0.360 | 1.6e-175 |
| MPNet-base | 0.0312 | 0.0123 | 0.380 | 0.367-0.393 | 3.0e-192 |
| mxbai-large | 0.0323 | 0.0144 | 0.419 | 0.404-0.434 | 1.4e-174 |
| GTE-base | 0.0249 | 0.0120 | 0.426 | 0.406-0.446 | 9.1e-136 |
| Qwen3-0.6B | 0.0834 | 0.0506 | 0.442 | 0.423-0.462 | 8.5e-139 |
| Qwen3-4B | 0.0605 | 0.0387 | 0.602 | 0.583-0.623 | 3.0e-129 |
| Qwen3-8B | 0.0731 | 0.0715 | 0.889 | 0.858-0.922 | 5.5e-44 |

Passive contrast, verb held fixed. Distance = 1 - centred cosine. A ratio below 1 means the sentence that reverses who acts on whom sits closer than the one that preserves the meaning in other words.


## T01b. The scale trend is not an artefact of precision
*Source: e2 minimal-pairs*

| model | precision | passive ratio | 90% CI | twin ratio | 90% CI |
|---|---|---|---|---|---|
| Qwen3-0.6B | fp16 | 0.442 | 0.423-0.462 | 0.705 | 0.666-0.746 |
| Qwen3-0.6B | fp32 | 0.442 | 0.423-0.462 | 0.705 | 0.666-0.746 |
| Qwen3-0.6B | int8 | 0.465 | 0.447-0.485 | 0.723 | 0.686-0.761 |
| Qwen3-4B | fp16 | 0.602 | 0.583-0.623 | 0.783 | 0.746-0.822 |
| Qwen3-4B | int8 | 0.635 | 0.616-0.654 | 0.800 | 0.765-0.838 |
| Qwen3-8B | int8 | 0.889 | 0.858-0.922 | 0.989 | 0.946-1.033 |

Sizes in a family ship at different default precisions and quantisation drives the ratio toward 1, in the same direction as the trend with size, so the two are crossed rather than left at each default. Compare down a column, at a common precision.


## T02. Equivalence tests on the verb contrast
*Source: e2 minimal-pairs*

| model | kept cos | reversed cos | ratio | 90% CI | P not further | P alike | role-blind |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.9270 | 0.9441 | 0.705 | 0.666-0.746 | 3.5e-48 | 1.0e+00 | yes |
| mxbai-large | 0.9806 | 0.9857 | 0.714 | 0.667-0.765 | 1.8e-34 | 1.0e+00 | yes |
| Qwen3-4B | 0.9431 | 0.9519 | 0.783 | 0.746-0.822 | 3.0e-44 | 7.6e-01 | yes |
| GTE-base | 0.9853 | 0.9871 | 0.838 | 0.782-0.897 | 3.2e-20 | 1.3e-01 | yes |
| sup-SimCSE | 0.9894 | 0.9913 | 0.869 | 0.821-0.920 | 4.9e-23 | 9.0e-03 | yes |
| E5-base | 0.9885 | 0.9889 | 0.879 | 0.828-0.932 | 1.1e-20 | 4.8e-03 | yes |
| Qwen3-8B | 0.9352 | 0.9276 | 0.989 | 0.946-1.033 | 2.5e-17 | 1.1e-14 | yes |
| MPNet-base | 0.9860 | 0.9851 | 1.012 | 0.963-1.063 | 3.3e-12 | 3.3e-12 | yes |
| BGE-base | 0.9901 | 0.9894 | 1.037 | 0.979-1.099 | 8.9e-08 | 8.9e-08 | yes |
| retrained | 0.9952 | 0.9949 | 1.063 | 1.038-1.090 | 9.3e-25 | 9.3e-25 | yes |
| unsup-SimCSE | 0.9905 | 0.9891 | 1.095 | 1.059-1.133 | 1.6e-10 | 1.6e-10 | yes |

Bounds 0.80-1.25, fixed before any model was run (the bioequivalence convention). P alike tests the stricter claim that the two are treated the same, not merely that the reversal is not further. Cosines, not distances: the contrast is recorded as cosine in the protocol.


## T02b. Per-cell evidence, predicate excluded
*Source: e2 minimal-pairs*

| model | reversed / kept | 90% CI | P not further | P alike | role-blind |
|---|---|---|---|---|---|
| unsup-SimCSE | 1.026 | 1.002-1.051 | 2.9e-35 | 2.9e-35 | yes |

The same equivalence test applied to the explanation's own evidence rather than to the similarity. It replaces the withdrawn half-strength claim: the evidence for a reversed pair is not weaker, it is the same.


## T03. A linear probe reads the roles off the representation
*Source: e2 probe*

| features | act->act | act->pas | both->act | both->pas | control seen | control new |
|---|---|---|---|---|---|---|
| calibration: bag of names | 50.0% | 50.0% | 50.0% | 50.0% | 100.0% | 55.1% |
| calibration: names by order | 100.0% | 0.0% | 50.0% | 50.0% | 100.0% | 53.2% |
| calibration: names by role | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 53.2% |
| unsup-SimCSE | 76.0% | 26.1% | 65.0% | 68.0% | 93.3% | 52.3% |
| sup-SimCSE | 75.9% | 29.3% | 66.8% | 67.3% | 92.4% | 53.7% |
| BGE-base | 83.9% | 26.9% | 75.1% | 72.9% | 95.8% | 50.7% |
| E5-base | 81.6% | 23.2% | 72.1% | 68.8% | 95.7% | 52.1% |
| retrained | 68.0% | 34.0% | 60.8% | 60.8% | 95.2% | 53.1% |
| Qwen3-0.6B | 98.2% | 1.1% | 80.6% | 75.6% | 95.8% | 53.1% |
| GTE-base | 82.9% | 21.8% | 75.2% | 70.6% | 97.8% | 50.6% |
| MPNet-base | 83.1% | 22.6% | 76.1% | 72.6% | 93.3% | 53.0% |
| mxbai-large | 84.2% | 24.2% | 75.8% | 71.3% | 93.9% | 50.9% |
| Qwen3-4B | 98.5% | 1.1% | 81.2% | 74.4% | 95.6% | 50.7% |
| Qwen3-8B | 99.2% | 5.4% | 83.8% | 66.6% | 93.9% | 52.2% |

Tested on name pairs never seen in training. An order code alone cannot exceed 50% on both voices, so both->act and both->pas above 50% together mean names are bound to roles. The control task (Hewitt-Liang, random labels) near 50% excludes memorisation.


## T04. No layer's similarity separates a role reversal
*Source: e2 layers*

| model | layer | probe active | probe passive | distance ratio | 90% CI |
|---|---|---|---|---|---|
| unsup-SimCSE | 1 | 54.3% | 56.7% | 0.019 | 0.018-0.020 |
| unsup-SimCSE | 2 | 60.9% | 68.6% | 0.010 | 0.009-0.010 |
| unsup-SimCSE | 3 | 63.2% | 72.6% | 0.011 | 0.011-0.011 |
| unsup-SimCSE | 4 | 69.4% | 70.1% | 0.042 | 0.041-0.043 |
| unsup-SimCSE | 5 | 74.7% | 70.6% | 0.099 | 0.096-0.103 |
| unsup-SimCSE | 6 | 75.2% | 70.7% | 0.087 | 0.084-0.090 |
| unsup-SimCSE | 7 | 74.4% | 70.8% | 0.095 | 0.092-0.098 |
| unsup-SimCSE | 8 | 71.7% | 68.8% | 0.059 | 0.057-0.060 |
| unsup-SimCSE | 9 | 74.4% | 63.4% | 0.059 | 0.058-0.061 |
| unsup-SimCSE | 10 | 64.7% | 64.9% | 0.066 | 0.064-0.069 |
| unsup-SimCSE | 11 | 62.3% | 65.8% | 0.143 | 0.137-0.149 |
| unsup-SimCSE | 12 | 63.4% | 64.3% | 0.184 | 0.177-0.191 |
| Qwen3-0.6B | 1 | 55.6% | 71.7% | 0.009 | 0.007-0.010 |
| Qwen3-0.6B | 2 | 73.2% | 84.3% | 0.066 | 0.061-0.072 |
| Qwen3-0.6B | 3 | 76.1% | 82.1% | 0.085 | 0.079-0.092 |
| Qwen3-0.6B | 4 | 74.6% | 79.5% | 0.103 | 0.096-0.111 |
| Qwen3-0.6B | 5 | 77.8% | 79.4% | 0.097 | 0.090-0.104 |
| Qwen3-0.6B | 6 | 76.3% | 80.1% | 0.098 | 0.092-0.105 |
| Qwen3-0.6B | 7 | 76.2% | 79.4% | 0.102 | 0.095-0.109 |
| Qwen3-0.6B | 8 | 76.2% | 77.6% | 0.149 | 0.140-0.158 |
| Qwen3-0.6B | 9 | 75.4% | 74.5% | 0.139 | 0.130-0.148 |
| Qwen3-0.6B | 10 | 74.4% | 76.2% | 0.157 | 0.148-0.167 |
| Qwen3-0.6B | 11 | 75.3% | 75.2% | 0.154 | 0.145-0.163 |
| Qwen3-0.6B | 12 | 74.7% | 74.7% | 0.161 | 0.152-0.170 |
| Qwen3-0.6B | 13 | 74.3% | 77.0% | 0.168 | 0.159-0.177 |
| Qwen3-0.6B | 14 | 79.4% | 79.6% | 0.167 | 0.158-0.176 |
| Qwen3-0.6B | 15 | 79.0% | 81.9% | 0.166 | 0.158-0.175 |
| Qwen3-0.6B | 16 | 77.0% | 81.1% | 0.175 | 0.167-0.185 |
| Qwen3-0.6B | 17 | 80.4% | 82.8% | 0.177 | 0.167-0.186 |
| Qwen3-0.6B | 18 | 80.4% | 83.5% | 0.179 | 0.171-0.188 |
| Qwen3-0.6B | 19 | 78.6% | 82.7% | 0.183 | 0.175-0.193 |
| Qwen3-0.6B | 20 | 79.8% | 81.0% | 0.190 | 0.182-0.200 |
| Qwen3-0.6B | 21 | 79.9% | 77.8% | 0.197 | 0.188-0.207 |
| Qwen3-0.6B | 22 | 80.3% | 68.4% | 0.295 | 0.280-0.310 |
| Qwen3-0.6B | 23 | 78.1% | 71.4% | 0.302 | 0.288-0.316 |
| Qwen3-0.6B | 24 | 77.2% | 73.7% | 0.303 | 0.290-0.318 |
| Qwen3-0.6B | 25 | 77.8% | 70.0% | 0.352 | 0.336-0.369 |
| Qwen3-0.6B | 26 | 75.1% | 67.1% | 0.419 | 0.400-0.438 |
| Qwen3-0.6B | 27 | 76.8% | 68.8% | 0.443 | 0.424-0.462 |
| Qwen3-0.6B | 28 | 76.8% | 67.3% | 0.442 | 0.423-0.462 |

Layer 0 is a constant representation under CLS pooling, so no distance exists there and it is omitted. A ratio above 1 at any layer would mean reading that layer repairs the failure without retraining.


## T05. Explanation measures, by dataset
*Source: e2 measures*

| dataset | model | n | cosine | cross-position share | per-cell ratio | overlap mass | mass vs score | faithfulness |
|---|---|---|---|---|---|---|---|---|
| mrpc | BGE-base | [200, 200] | 0.854/0.763 | 0.267/0.429 | 0.36/0.76 | 0.282/0.259 | -- | 98% |
| mrpc | E5-base | [200, 200] | 0.859/0.771 | 0.265/0.431 | 0.39/0.74 | 0.271/0.255 | -- | 99% |
| mrpc | Qwen3-0.6B | [200, 200] | 0.850/0.779 | 0.211/0.297 | 0.08/0.14 | 0.347/0.335 | -- | 94% |
| mrpc | retrained | [200, 200] | 0.839/0.751 | 0.265/0.434 | 0.43/0.76 | 0.170/0.171 | -- | 92% |
| mrpc | sup-SimCSE | [200, 200] | 0.864/0.787 | 0.274/0.429 | 0.38/0.63 | 0.223/0.211 | -- | 98% |
| mrpc | unsup-SimCSE | [200, 200] | 0.838/0.741 | 0.263/0.434 | 0.44/0.72 | 0.152/0.148 | -- | 95% |
| paws | BGE-base | [400, 400] | 0.963/0.958 | 0.177/0.311 | 0.26/0.79 | 0.359/0.360 | -- | 97% |
| paws | E5-base | [400, 400] | 0.968/0.963 | 0.180/0.293 | 0.32/0.73 | 0.310/0.312 | -- | 97% |
| paws | Qwen3-0.6B | [400, 400] | 0.959/0.932 | 0.102/0.188 | 0.06/0.15 | 0.466/0.448 | -- | 94% |
| paws | retrained | [400, 400] | 0.971/0.968 | 0.192/0.311 | 0.34/0.77 | 0.231/0.229 | -- | 93% |
| paws | sup-SimCSE | [400, 400] | 0.974/0.969 | 0.185/0.303 | 0.32/0.71 | 0.274/0.273 | -- | 99% |
| paws | unsup-SimCSE | [400, 400] | 0.961/0.957 | 0.184/0.283 | 0.32/0.65 | 0.226/0.225 | -- | 97% |
| stsb | BGE-base | 300 | 0.543 | 0.322 | 0.58 | 0.180 | 0.645 | 90% |
| stsb | E5-base | 300 | 0.560 | 0.317 | 0.55 | 0.182 | 0.639 | 94% |
| stsb | sup-SimCSE | 300 | 0.548 | 0.336 | 0.66 | 0.216 | 0.515 | 94% |
| stsb | unsup-SimCSE | 300 | 0.546 | 0.364 | 0.79 | 0.159 | 0.534 | 95% |
| swapmin | Qwen3-0.6B | [400, 400] | 0.928/0.945 | 0.237/0.239 | 0.63/0.67 | 0.460/0.460 | -- | 68% |
| swapmin | sup-SimCSE | [400, 400] | 0.989/0.991 | 0.171/0.194 | 0.76/0.91 | 0.568/0.573 | -- | 90% |

Where two values are separated by a slash they are class 1 / class 0 (paraphrase or meaning kept / non-paraphrase or roles reversed). STS-B is graded rather than classed, so its values are over all pairs and the mass-vs-score column is how far identical-word evidence tracks the human score. NOT the same quantity as the partial-Spearman table (F6), which is still legacy output and needs the e1 rerun.


## T06. Graded supervision and its shuffled control
*Source: train graded*

| condition | seed | STS-B | overlap excess | fooled | queries fixed / broken | McNemar P |
|---|---|---|---|---|---|---|
| base | -- | 0.818 | +0.510 | 0.276 | -- | -- |
| real scores | 42 | 0.859 | +0.387 | 0.220 | 48 / 20 | 9.1e-04 |
| real scores | 1 | 0.862 | +0.399 | 0.232 | 45 / 23 | 1.0e-02 |
| real scores | 2 | 0.866 | +0.392 | 0.234 | 41 / 20 | 9.9e-03 |
| shuffled | 42 | 0.628 | +0.651 | 0.250 | 54 / 41 | 2.2e-01 |
| shuffled | 1 | 0.595 | +0.486 | 0.220 | 56 / 28 | 3.0e-03 |
| shuffled | 2 | 0.608 | +0.529 | 0.246 | 64 / 49 | 1.9e-01 |

Identical data, steps and schedule; the control shuffles the scores. Separated across seeds: the excess and STS-B. NOT separated: the fooled rate. Do not state a causal link from removing the excess to fixing retrieval.


## T07. What the negative has to be
*Source: train negatives*

| negative | seed | its overlap with the anchor | STS-B | overlap excess | fooled |
|---|---|---|---|---|---|
| base | -- | -- | 0.818 | +0.510 | 0.276 |
| none | 1 | 0.000 | 0.801 | +0.510 | 0.294 |
| none | 2 | 0.000 | 0.803 | +0.509 | 0.274 |
| random | 1 | 0.005 | 0.799 | +0.507 | 0.292 |
| random | 2 | 0.005 | 0.797 | +0.507 | 0.272 |
| topic | 1 | 0.137 | 0.779 | +0.503 | 0.254 |
| topic | 2 | 0.137 | 0.783 | +0.503 | 0.248 |
| shuffled | 1 | 1.000 | 0.706 | +0.502 | 0.260 |
| shuffled | 2 | 1.000 | 0.715 | +0.511 | 0.266 |
| non-role swap | 1 | 1.000 | 0.647 | +0.457 | 0.296 |
| non-role swap | 2 | 1.000 | 0.645 | +0.459 | 0.304 |
| role swap | 1 | 1.000 | 0.729 | +0.317 | 0.150 |
| role swap | 2 | 1.000 | 0.726 | +0.337 | 0.134 |

If word matching survives because nothing in training punishes it, a negative that shares the anchor's words should be the one that removes it.


## T08. Retrieval and benchmark scores per checkpoint
*Source: train evaluate*

| checkpoint | top-1 clean | top-1 with swap | fooled | STS-B | overlap excess |
|---|---|---|---|---|---|
| retrained | 1.000 | 0.726 | 0.276 | 0.818 | +0.510 |
| ckpt/retrained_stsb_real_s42 | 1.000 | 0.780 | 0.220 | 0.859 | +0.387 |
| ckpt/retrained_stsb_real_s1 | 1.000 | 0.766 | 0.232 | 0.862 | +0.399 |
| ckpt/retrained_stsb_real_s2 | 1.000 | 0.766 | 0.234 | 0.866 | +0.392 |
| ckpt/retrained_stsb_shuffled_s42 | 1.000 | 0.750 | 0.250 | 0.628 | +0.651 |
| ckpt/retrained_stsb_shuffled_s1 | 1.000 | 0.780 | 0.220 | 0.595 | +0.486 |
| ckpt/retrained_stsb_shuffled_s2 | 1.000 | 0.754 | 0.246 | 0.608 | +0.529 |

A fooled rate that falls while top-1 clean also falls is a worse retriever, not a repaired one. Read the two columns together.


## T09. Roles are encoded per entity, not on a shared axis
*Source: e4 entanglement*

| model | names | role vs role | random level | own name | other names | variance in top directions |
|---|---|---|---|---|---|---|
| unsup-SimCSE | 40 | 0.090 | 0.030 | 0.681 | 0.087 | 53% |
| retrained | 40 | 0.096 | 0.030 | 0.527 | 0.089 | 54% |
| Qwen3-0.6B | 40 | 0.091 | 0.025 | 0.541 | 0.056 | 55% |

Role direction of a name = that name as agent minus that name as patient. Role directions align with each other barely above the random level, and each rides on its own name's identity direction. An edit would need a different direction for every entity.


## T10. What separating role swaps costs
*Source: e4 entanglement*

| model | edit | rank | lambda | similarity removed | STS-B lost | STS-B per 0.01 removed |
|---|---|---|---|---|---|---|
| retrained | control swaps | 768 | 1000 | 0.0036 | 0.0002 | 0.000 |
| retrained | control swaps | 768 | 300 | 0.0052 | 0.0012 | 0.002 |
| retrained | control swaps | 768 | 100 | 0.0138 | 0.0103 | 0.007 |
| retrained | control swaps | 768 | 30 | 0.0390 | 0.0558 | 0.014 |
| retrained | entity swaps | 768 | 30 | 0.0032 | 0.0109 | 0.034 |
| retrained | control swaps | 768 | 10 | 0.0926 | 0.1611 | 0.017 |
| retrained | entity swaps | 768 | 10 | 0.0118 | 0.0615 | 0.052 |
| retrained | control swaps | 64 | 1000 | 0.0044 | 0.0006 | 0.001 |
| retrained | control swaps | 64 | 300 | 0.0089 | 0.0035 | 0.004 |
| retrained | control swaps | 64 | 100 | 0.0190 | 0.0180 | 0.009 |
| retrained | control swaps | 64 | 30 | 0.0364 | 0.0524 | 0.014 |
| retrained | entity swaps | 64 | 30 | 0.0054 | 0.0204 | 0.038 |
| retrained | control swaps | 64 | 10 | 0.0652 | 0.1191 | 0.018 |
| retrained | entity swaps | 64 | 10 | 0.0112 | 0.0555 | 0.050 |
| unsup-SimCSE | control swaps | 768 | 1000 | 0.0072 | 0.0001 | 0.000 |
| unsup-SimCSE | control swaps | 768 | 300 | 0.0105 | 0.0031 | 0.003 |
| unsup-SimCSE | control swaps | 768 | 100 | 0.0250 | 0.0236 | 0.009 |
| unsup-SimCSE | entity swaps | 768 | 100 | 0.0034 | 0.0057 | 0.017 |
| unsup-SimCSE | control swaps | 768 | 30 | 0.0680 | 0.1132 | 0.017 |
| unsup-SimCSE | entity swaps | 768 | 30 | 0.0166 | 0.0400 | 0.024 |
| unsup-SimCSE | control swaps | 768 | 10 | 0.1398 | 0.2118 | 0.015 |
| unsup-SimCSE | entity swaps | 768 | 10 | 0.0508 | 0.1322 | 0.026 |
| unsup-SimCSE | control swaps | 64 | 1000 | 0.0089 | 0.0015 | 0.002 |
| unsup-SimCSE | control swaps | 64 | 300 | 0.0172 | 0.0116 | 0.007 |
| unsup-SimCSE | control swaps | 64 | 100 | 0.0335 | 0.0425 | 0.013 |
| unsup-SimCSE | entity swaps | 64 | 100 | 0.0079 | 0.0148 | 0.019 |
| unsup-SimCSE | control swaps | 64 | 30 | 0.0594 | 0.1018 | 0.017 |
| unsup-SimCSE | entity swaps | 64 | 30 | 0.0208 | 0.0531 | 0.026 |
| unsup-SimCSE | control swaps | 64 | 10 | 0.0988 | 0.1688 | 0.017 |
| unsup-SimCSE | entity swaps | 64 | 10 | 0.0364 | 0.0985 | 0.027 |

Compare rows only at the same rank and lambda: each lambda removes a different amount, so a rate averaged across them is not a like-for-like comparison. Read with T09 and T14: the rank-64 edit lacked the capacity to span the binding, and the rank-768 edit had it and failed on cost instead.


## T11. What the correction costs zero-shot retrieval
*Source: e3 nanobeir*

| condition | mean | change | fooled |
|---|---|---|---|
| retrained | 0.384 | -- | -- |
| ckpt/retrained_neg_shuffled_s1 | 0.260 | -- | -- |
| ckpt/retrained_neg_shuffled_s2 | 0.266 | -- | -- |
| ckpt/retrained_neg_non-role-swap_s1 | 0.235 | -- | -- |
| ckpt/retrained_neg_non-role-swap_s2 | 0.239 | -- | -- |

NanoBEIR nDCG@10, all conditions trained identically. One seed each except where the caption gives three. The benefit is stable across seeds while the cost varies by a factor of two and a half, so the damage looks incidental rather than intrinsic.


## T12. The binding term, decomposed
*Source: e5 decompose*

| model | |D|/|S| | variance share | across pairs | random level | own entity | other entities |
|---|---|---|---|---|---|---|
| unsup-SimCSE | 0.100 | 0.010 | 0.147 | 0.029 | 0.288 | 0.086 |
| BGE-base | 0.121 | 0.015 | 0.111 | 0.029 | 0.474 | 0.057 |
| Qwen3-0.6B | 0.277 | 0.072 | 0.087 | 0.025 | 0.652 | 0.066 |

Fixed frame, verb, year and city throughout, so the entity pair is the only thing that varies. Quote absolute cosines from the varied-frame run instead: centring differs between the two and they are not comparable.


## T13. The rank of the binding term grows with the vocabulary
*Source: e5 decompose*

| model | 30 entities | 60 entities | 90 entities | 120 entities |
|---|---|---|---|---|
| a generic role axis would need | 1 | 1 | 1 | 1 |
| unsup-SimCSE | 37 | 55 | 69 | 78 |
| BGE-base | 26 | 46 | 62 | 75 |
| Qwen3-0.6B | 23 | 41 | 56 | 68 |

Directions holding 90% of the binding term, at a fixed item count (ceiling 435 in every column, so the rank is nowhere near the sample bound). Growth is sub-linear and will saturate; we cannot say where. Do not extrapolate to the embedding dimension.


## T14. H1  HANS entailment
*Source: e6 blind*

| model | overall AUC | 90% CI | ln_passive | ln_subject/object_swap |
|---|---|---|---|---|
| unsup-SimCSE | 0.322 | 0.306-0.339 | 0.558 | 0.087 |
| sup-SimCSE | 0.369 | 0.351-0.387 | 0.519 | 0.219 |
| BGE-base | 0.321 | 0.305-0.336 | 0.501 | 0.141 |
| E5-base | 0.572 | 0.553-0.589 | 0.656 | 0.489 |
| retrained | 0.292 | 0.278-0.308 | 0.549 | 0.036 |

H1: DOES NOT HOLD -- registered: every English encoder between 0.40 and 0.60 AUC. AUC is the probability a true entailment scores above a non-entailment. We predicted indifference; below 0.5 the similarity prefers the role-swapped non-entailment. Report the miss as a miss: the effect exceeded the prediction.


## T18. Every worked example, every model
*Source: fig_examples*

| domain | model | original | same meaning / new words | opposite meaning / same words | d kept | d reversed | ratio | closer |
|---|---|---|---|---|---|---|---|---|
| pharmacology | BGE-base | Ritonavir inhibits the metabolism of midazolam. | The metabolism of midazolam is slowed by ritonavir. | Midazolam inhibits the metabolism of ritonavir. | 0.0987 | 0.0243 | 0.247 | reversal |
| pharmacology | E5-base | Ritonavir inhibits the metabolism of midazolam. | The metabolism of midazolam is slowed by ritonavir. | Midazolam inhibits the metabolism of ritonavir. | 0.0863 | 0.0165 | 0.191 | reversal |
| pharmacology | Qwen3-0.6B | Ritonavir inhibits the metabolism of midazolam. | The metabolism of midazolam is slowed by ritonavir. | Midazolam inhibits the metabolism of ritonavir. | 0.1211 | 0.0982 | 0.811 | reversal |
| pharmacology | Qwen3-8B | Ritonavir inhibits the metabolism of midazolam. | The metabolism of midazolam is slowed by ritonavir. | Midazolam inhibits the metabolism of ritonavir. | 0.2303 | 0.1158 | 0.503 | reversal |
| pharmacology | unsup-SimCSE | Ritonavir inhibits the metabolism of midazolam. | The metabolism of midazolam is slowed by ritonavir. | Midazolam inhibits the metabolism of ritonavir. | 0.3153 | 0.0163 | 0.052 | reversal |
| pharmacology | BGE-base | Rifampicin induces the enzyme that clears warfarin. | Warfarin is cleared faster because rifampicin induces the enzyme. | Warfarin induces the enzyme that clears rifampicin. | 0.1413 | 0.0160 | 0.113 | reversal |
| pharmacology | E5-base | Rifampicin induces the enzyme that clears warfarin. | Warfarin is cleared faster because rifampicin induces the enzyme. | Warfarin induces the enzyme that clears rifampicin. | 0.1479 | 0.0184 | 0.124 | reversal |
| pharmacology | Qwen3-0.6B | Rifampicin induces the enzyme that clears warfarin. | Warfarin is cleared faster because rifampicin induces the enzyme. | Warfarin induces the enzyme that clears rifampicin. | 0.1693 | 0.0992 | 0.586 | reversal |
| pharmacology | Qwen3-8B | Rifampicin induces the enzyme that clears warfarin. | Warfarin is cleared faster because rifampicin induces the enzyme. | Warfarin induces the enzyme that clears rifampicin. | 0.1804 | 0.1182 | 0.655 | reversal |
| pharmacology | unsup-SimCSE | Rifampicin induces the enzyme that clears warfarin. | Warfarin is cleared faster because rifampicin induces the enzyme. | Warfarin induces the enzyme that clears rifampicin. | 0.2998 | 0.0146 | 0.049 | reversal |
| epidemiology | BGE-base | The virus was transmitted from the patient to the nurse. | The nurse caught the virus from the patient. | The virus was transmitted from the nurse to the patient. | 0.1773 | 0.0036 | 0.020 | reversal |
| epidemiology | E5-base | The virus was transmitted from the patient to the nurse. | The nurse caught the virus from the patient. | The virus was transmitted from the nurse to the patient. | 0.1100 | 0.0022 | 0.020 | reversal |
| epidemiology | Qwen3-0.6B | The virus was transmitted from the patient to the nurse. | The nurse caught the virus from the patient. | The virus was transmitted from the nurse to the patient. | 0.1532 | 0.0203 | 0.132 | reversal |
| epidemiology | Qwen3-8B | The virus was transmitted from the patient to the nurse. | The nurse caught the virus from the patient. | The virus was transmitted from the nurse to the patient. | 0.1686 | 0.0451 | 0.268 | reversal |
| epidemiology | unsup-SimCSE | The virus was transmitted from the patient to the nurse. | The nurse caught the virus from the patient. | The virus was transmitted from the nurse to the patient. | 0.2259 | 0.0035 | 0.015 | reversal |
| molecular biology | BGE-base | The kinase phosphorylates the receptor. | The receptor is phosphorylated by the kinase. | The receptor phosphorylates the kinase. | 0.0763 | 0.0304 | 0.399 | reversal |
| molecular biology | E5-base | The kinase phosphorylates the receptor. | The receptor is phosphorylated by the kinase. | The receptor phosphorylates the kinase. | 0.0660 | 0.0167 | 0.254 | reversal |
| molecular biology | Qwen3-0.6B | The kinase phosphorylates the receptor. | The receptor is phosphorylated by the kinase. | The receptor phosphorylates the kinase. | 0.0634 | 0.0493 | 0.777 | reversal |
| molecular biology | Qwen3-8B | The kinase phosphorylates the receptor. | The receptor is phosphorylated by the kinase. | The receptor phosphorylates the kinase. | 0.0681 | 0.1488 | 2.186 | paraphrase |
| molecular biology | unsup-SimCSE | The kinase phosphorylates the receptor. | The receptor is phosphorylated by the kinase. | The receptor phosphorylates the kinase. | 0.0595 | 0.0047 | 0.080 | reversal |
| law | BGE-base | The plaintiff sued the defendant for breach of contract. | The defendant was sued by the plaintiff for breach of contract. | The defendant sued the plaintiff for breach of contract. | 0.0291 | 0.0112 | 0.383 | reversal |
| law | E5-base | The plaintiff sued the defendant for breach of contract. | The defendant was sued by the plaintiff for breach of contract. | The defendant sued the plaintiff for breach of contract. | 0.0215 | 0.0049 | 0.229 | reversal |
| law | Qwen3-0.6B | The plaintiff sued the defendant for breach of contract. | The defendant was sued by the plaintiff for breach of contract. | The defendant sued the plaintiff for breach of contract. | 0.0604 | 0.0235 | 0.390 | reversal |
| law | Qwen3-8B | The plaintiff sued the defendant for breach of contract. | The defendant was sued by the plaintiff for breach of contract. | The defendant sued the plaintiff for breach of contract. | 0.0714 | 0.2164 | 3.032 | paraphrase |
| law | unsup-SimCSE | The plaintiff sued the defendant for breach of contract. | The defendant was sued by the plaintiff for breach of contract. | The defendant sued the plaintiff for breach of contract. | 0.0192 | 0.0010 | 0.053 | reversal |
| clinical | BGE-base | The donor's kidney was transplanted into the recipient. | The recipient received a kidney from the donor. | The recipient's kidney was transplanted into the donor. | 0.1109 | 0.0203 | 0.183 | reversal |
| clinical | E5-base | The donor's kidney was transplanted into the recipient. | The recipient received a kidney from the donor. | The recipient's kidney was transplanted into the donor. | 0.0930 | 0.0095 | 0.103 | reversal |
| clinical | Qwen3-0.6B | The donor's kidney was transplanted into the recipient. | The recipient received a kidney from the donor. | The recipient's kidney was transplanted into the donor. | 0.1495 | 0.0474 | 0.317 | reversal |
| clinical | Qwen3-8B | The donor's kidney was transplanted into the recipient. | The recipient received a kidney from the donor. | The recipient's kidney was transplanted into the donor. | 0.1438 | 0.3733 | 2.596 | paraphrase |
| clinical | unsup-SimCSE | The donor's kidney was transplanted into the recipient. | The recipient received a kidney from the donor. | The recipient's kidney was transplanted into the donor. | 0.1732 | 0.0027 | 0.016 | reversal |
| news | BGE-base | Germany defeated Brazil in the semi-final. | Brazil lost to Germany in the semi-final. | Brazil defeated Germany in the semi-final. | 0.1458 | 0.0139 | 0.095 | reversal |
| news | E5-base | Germany defeated Brazil in the semi-final. | Brazil lost to Germany in the semi-final. | Brazil defeated Germany in the semi-final. | 0.0940 | 0.0393 | 0.418 | reversal |
| news | Qwen3-0.6B | Germany defeated Brazil in the semi-final. | Brazil lost to Germany in the semi-final. | Brazil defeated Germany in the semi-final. | 0.1365 | 0.1002 | 0.734 | reversal |
| news | Qwen3-8B | Germany defeated Brazil in the semi-final. | Brazil lost to Germany in the semi-final. | Brazil defeated Germany in the semi-final. | 0.1691 | 0.1861 | 1.100 | paraphrase |
| news | unsup-SimCSE | Germany defeated Brazil in the semi-final. | Brazil lost to Germany in the semi-final. | Brazil defeated Germany in the semi-final. | 0.0911 | 0.0075 | 0.083 | reversal |
| finance | BGE-base | Barclays acquired Lehman's brokerage arm. | Lehman's brokerage arm was bought by Barclays. | Lehman acquired Barclays' brokerage arm. | 0.0561 | 0.0100 | 0.179 | reversal |
| finance | E5-base | Barclays acquired Lehman's brokerage arm. | Lehman's brokerage arm was bought by Barclays. | Lehman acquired Barclays' brokerage arm. | 0.0382 | 0.0092 | 0.241 | reversal |
| finance | Qwen3-0.6B | Barclays acquired Lehman's brokerage arm. | Lehman's brokerage arm was bought by Barclays. | Lehman acquired Barclays' brokerage arm. | 0.1342 | 0.1447 | 1.078 | paraphrase |
| finance | Qwen3-8B | Barclays acquired Lehman's brokerage arm. | Lehman's brokerage arm was bought by Barclays. | Lehman acquired Barclays' brokerage arm. | 0.1054 | 0.1070 | 1.014 | paraphrase |
| finance | unsup-SimCSE | Barclays acquired Lehman's brokerage arm. | Lehman's brokerage arm was bought by Barclays. | Lehman acquired Barclays' brokerage arm. | 0.0598 | 0.0177 | 0.297 | reversal |
| negation | BGE-base | The trial showed the drug reduces mortality. | Mortality fell in the group given the drug. | The trial showed the drug does not reduce mortality. | 0.4501 | 0.2289 | 0.508 | reversal |
| negation | E5-base | The trial showed the drug reduces mortality. | Mortality fell in the group given the drug. | The trial showed the drug does not reduce mortality. | 0.3828 | 0.2357 | 0.616 | reversal |
| negation | Qwen3-0.6B | The trial showed the drug reduces mortality. | Mortality fell in the group given the drug. | The trial showed the drug does not reduce mortality. | 0.3153 | 0.2888 | 0.916 | reversal |
| negation | Qwen3-8B | The trial showed the drug reduces mortality. | Mortality fell in the group given the drug. | The trial showed the drug does not reduce mortality. | 0.2456 | 0.3368 | 1.372 | paraphrase |
| negation | unsup-SimCSE | The trial showed the drug reduces mortality. | Mortality fell in the group given the drug. | The trial showed the drug does not reduce mortality. | 0.5138 | 0.0677 | 0.132 | reversal |
| quantity | BGE-base | The dose was increased from 5 mg to 20 mg. | The dose went up to 20 mg from 5 mg. | The dose was increased from 20 mg to 5 mg. | 0.0531 | 0.0019 | 0.035 | reversal |
| quantity | E5-base | The dose was increased from 5 mg to 20 mg. | The dose went up to 20 mg from 5 mg. | The dose was increased from 20 mg to 5 mg. | 0.0616 | 0.0024 | 0.038 | reversal |
| quantity | Qwen3-0.6B | The dose was increased from 5 mg to 20 mg. | The dose went up to 20 mg from 5 mg. | The dose was increased from 20 mg to 5 mg. | 0.0637 | 0.0228 | 0.357 | reversal |
| quantity | Qwen3-8B | The dose was increased from 5 mg to 20 mg. | The dose went up to 20 mg from 5 mg. | The dose was increased from 20 mg to 5 mg. | 0.1030 | 0.0883 | 0.857 | reversal |
| quantity | unsup-SimCSE | The dose was increased from 5 mg to 20 mg. | The dose went up to 20 mg from 5 mg. | The dose was increased from 20 mg to 5 mg. | 0.1001 | 0.0032 | 0.032 | reversal |

All items are reported, including those where the paraphrase correctly wins. The items were written before they were run and none was replaced after seeing a result.

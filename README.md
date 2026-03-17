# Continual Learning Atlas

**PDC-Powered Continual Knowledge Editing for Large Language Models**

[![arXiv](https://img.shields.io/badge/arXiv-preprint-blue)](https://arxiv.org/abs/XXXX)
[![MQuAKE](https://img.shields.io/badge/MQuAKE--CF--3k-84%25_Multi--Hop-brightgreen)]()
[![MMLU](https://img.shields.io/badge/MMLU-0.0%25_Degradation-brightgreen)]()

---

## Overview

The **Continual Learning Atlas** is a neural architecture for injecting thousands of isolated facts into a frozen LLM without any catastrophic forgetting.

| Result | Score |
|:---|:---:|
| MMLU Degradation (200 OOD facts ingested) | **0.0%** |
| MQuAKE-CF-3k Single-Hop Recall | **98.5%** |
| MQuAKE-CF-3k Multi-Hop Accuracy (PDC-ELQR) | **84.0%** |
| Standard LoRA MMLU Degradation (same task) | −4.0% |

---

## Key Ideas

- **EngramGatedChild Adapter** — each fact is a tiny (<2MB) isolated residual adapter, never merged into the base model weights.
- **Predictive Divergence Constraint (PDC)** — backpropagates only on tokens the base model finds surprising, creating pure entity injectors.
- **Sparse Keyword Router** — O(1) deterministic lookup. If the query doesn't overlap the adapter's entity vocabulary, the adapter stays 100% dormant.
- **Entity-Linked Query Routing (ELQR)** — PDC-extracted entities automatically build a knowledge graph for multi-hop chaining, zero manual NER required.

---

## Repository Structure

```
├── benchmark_mquake_cf_3k.py      # MQuAKE-CF-3k benchmark (98.5% / 84.0%)
├── benchmark_synthetic_mquake.py  # Synthetic MQuAKE benchmark
├── benchmark_mmlu_forgetting.py   # MMLU 0.0% degradation proof + LoRA comparison
├── Atlas_Research_Paper_Draft.md  # Full arXiv-style research paper
└── README.md                      # This file
```

---

## Quick Start (Google Colab T4 — Free)

```python
# Run the full MQuAKE-CF-3k benchmark
# Upload benchmark_mquake_cf_3k.py to Colab and run all cells
# Expected runtime: ~30 min on T4 for 100 cases
```

---

## Paper

Read the full paper: [`Atlas_Research_Paper_Draft.md`](./Atlas_Research_Paper_Draft.md)

**Author:** Adekoya Iyanuoluwa  
**arXiv preprint coming soon.**

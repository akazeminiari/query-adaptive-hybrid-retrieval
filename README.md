# Query-Adaptive Hybrid Retrieval Across Domains

Reproducibility materials for the manuscript:

**Query-Adaptive Hybrid Retrieval Across Domains: An Empirical Study of Effectiveness, Latency, and Protocol Sensitivity**

## Authors
- Ali Kazemi Niari (Corresponding Author)
- Asma Lesani

Department of Computer Engineering, Istanbul Rumeli University, Istanbul, Turkey.

## Overview
This study evaluates a frozen SciFact-trained eleven-feature router that selects between BM25-only retrieval and BM25–dense retrieval with Reciprocal Rank Fusion (RRF).

The experiments examine retrieval effectiveness, matched-budget random allocation, cross-domain transfer, CPU latency, and benchmark-protocol sensitivity.

## Repository Structure
- `src/` — Python evaluation and verification scripts
- `data/scifact/` — Archived SciFact router training data
- `results/` — Archived per-query effectiveness outputs
- `results/controlled_latency/` — Controlled CPU latency experiments
- `historical_audit/` — Historical ArguAna protocol-audit artifacts

## Main Experimental Settings
- Sparse retrieval: BM25Okapi
- Dense encoder: sentence-transformers/all-MiniLM-L6-v2
- Fusion: RRF, k = 10
- Ranking metric: nDCG@5
- Router: StandardScaler and LogisticRegression
- Frozen training source: SciFact (647 training queries)
- Dense-call randomization: 50,000 fixed-K allocations

## Reproducibility Scope
Archived query-level data support reproduction of reported statistical comparisons. Controlled latency files support recalculation of documented rank_bm25 and BM25S timing summaries.

The available materials do not establish complete end-to-end reproduction of every retrieval and original direct-timing experiment.

The original ArguAna evaluation included improper same-ID query–document matches. Its historical results are provided only for auditability and must not be interpreted as benchmark-compliant confirmatory evidence.

The script `run_confirmatory_beir.py` retains historical evaluation behavior. It should not be used to claim benchmark-compliant confirmation for ArguAna. Consult the manuscript and supplementary material for the corrected analysis.

The SciFact enrichment script depends on the original BEIR SciFact data and the archived router_dataset.csv. Some legacy scripts expect training CSV files under `outputs/router/`, whereas this repository stores the shared CSV files under `data/scifact/`.

## Data Sources
The benchmark datasets originate from BEIR. Original corpus files are not redistributed in this repository.

## Citation
If you use these materials, please cite the accompanying manuscript. Publication details will be added after acceptance.

## Contact
Ali Kazemi Niari
akazemi.niari@rumeli.edu.tr

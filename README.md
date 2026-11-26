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

The script `run_confirmatory_beir.py` is configured to evaluate TREC-COVID only. ArguAna is excluded from its default execution because the historical ArguAna evaluation contained improper same-ID query–document matches. Historical ArguAna artifacts are retained separately in `historical_audit/` for transparency and must not be interpreted as benchmark-compliant confirmatory results.

The available scripts and archived outputs support partial reproducibility. Full end-to-end reproduction of all original retrieval and latency experiments has not yet been established.

The SciFact enrichment script depends on the original BEIR SciFact data and the archived router_dataset.csv. Some legacy scripts expect training CSV files under `outputs/router/`, whereas this repository stores the shared CSV files under `data/scifact/`.

## Data Sources
The benchmark datasets originate from BEIR. Original corpus files are not redistributed in this repository.

## Citation
If you use these materials, please cite the accompanying manuscript. Publication details will be added after acceptance.

## Contact
Ali Kazemi Niari
akazemi.niari@rumeli.edu.tr

## Installation and Statistical Reproduction

Clone the repository:

```bash
git clone https://github.com/akazeminiari/query-adaptive-hybrid-retrieval.git
cd query-adaptive-hybrid-retrieval
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Reproduce the archived SCIDOCS and Android fixed-budget random allocation analyses:

```bash
python src/reproduce_allocation.py
```

The script reads the archived per-query results from `results/` and writes `src/SCIDOCS_Android_statistical_audit.csv`.

The TREC-COVID retrieval evaluation can be launched with:

```bash
python src/run_confirmatory_beir.py
```

This evaluation downloads the BEIR TREC-COVID dataset if necessary and runs the frozen SciFact-trained router. It is substantially more computationally demanding than the statistical reproduction script. ArguAna is excluded from the default evaluation.

The SciFact feature-validation script is available at `src/finalize_router_train_enriched.py` and requires the original BEIR SciFact training dataset under `datasets/scifact/`.

These instructions have not yet been independently validated through a complete end-to-end execution.

## License

The original source code authored by the researchers in
`src/` is made available under the MIT License.
See the LICENSE file.

The archived research datasets, query-level outputs,
historical audit files, and any third-party materials
are not automatically covered by this software license.
Their use remains subject to applicable data rights,
licenses, and attribution requirements.

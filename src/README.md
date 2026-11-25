# Source Code

Python scripts used in the retrieval, routing,
benchmark evaluation, and statistical analyses
described in the accompanying manuscript.

See the repository's main README for the
reproducibility scope and limitations.

### SciFact Router-11 Training Data Verification

The `finalize_router_train_enriched.py` script reconstructs the additional five BM25-derived features for the 647 SciFact training queries, verifies the original source feature values, and recomputes the frozen out-of-fold decision threshold.

The script requires the archived `router_dataset.csv` file and the original BEIR SciFact training data. It validates the consistency of the enriched training features and the frozen threshold but does not independently regenerate all original BM25 and Hybrid relevance labels from raw retrieval outputs.

The expected frozen threshold is 0.4994615425. The script terminates if the reconstructed threshold differs from the reference beyond the specified tolerance.

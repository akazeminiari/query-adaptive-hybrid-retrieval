# Controlled CPU Latency Experiments

This directory contains five-run controlled CPU latency
measurements for SCIDOCS and CQADupstack-Android,
comparing rank_bm25 and BM25S.

The original query-level routing decisions were held fixed.
These experiments are exploratory and do not reproduce
the initial three-repetition direct timing benchmarks.

The lexical engines may produce different rankings;
latency differences cannot be attributed to indexing alone.

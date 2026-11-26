# Reproducibility Verification

## Reference Release

GitHub release: v1.0.0

## Archived Statistical Analysis

The archived query-level outputs support fixed-budget
random-allocation analysis for SCIDOCS and
CQADupstack-Android.

The statistical reproduction script is:

`src/reproduce_allocation.py`

It uses 50,000 random allocations per dataset and
applies Holm correction across the two comparisons.

### Reported Results

| Dataset | Queries | Dense calls | Raw p | Holm p |
|---|---:|---:|---:|---:|
| SCIDOCS | 1000 | 654 | 0.201096 | 0.201096 |
| CQADupstack-Android | 699 | 227 | 0.001320 | 0.002640 |

These results concern statistical reproduction from
archived per-query outputs, not complete reruns of
the retrieval pipeline.

## Remaining Verification Work

- Full retrieval pipeline execution on TREC-COVID
- SciFact feature reconstruction from the BEIR corpus
- Independent reproduction of CPU latency experiments
- Verification of historical ArguAna audit artifacts
- Dependency-version and environment validation

## Protocol Limitations

Historical ArguAna results include an invalid
same-ID query-document matching protocol.
They are retained for auditability only and are
not confirmatory benchmark-compliant evidence.

## Scope

This repository supports partial reproducibility.
Complete end-to-end reproduction of all reported
experiments has not been established.

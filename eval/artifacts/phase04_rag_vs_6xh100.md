# Phase 04 RAG vs 6xH100 Baseline

- commit: `3de70ca`
- query: `GPU queue saturation and throughput metrics`
- RAG backend: `numpy-fallback`
- 6xH100 run: `modal-openai-server-1781943051`
- 6xH100 throughput: `1578.154` req/s, `91286.318` tok/s
- 6xH100 latency: p50 mean `45.765` ms, p95 max `182.296` ms
- 6xH100 derived MFU: `0.060593` of H100 BF16 peak
- RAG recall@5: `1.000000`; precision@5: `0.600000`; F1@5: `0.750000`
- Ours quality: recall `0.888889`; precision `1.000000`; F1 `0.941176`
- Ours recommended threshold: `0.01640298755872838` gives recall `0.888889`; precision `1.000000`; F1 `0.941176`

| docs | RAG index ms | RAG retrieve p50 ms | RAG retrieve qps | fresh-file total ms | retrieve/vLLM p50 | fresh/vLLM p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.111 | 0.047 | 21467.511 | 0.157 | 0.001x | 0.003x |
| 100 | 1.371 | 0.548 | 1825.927 | 1.918 | 0.012x | 0.042x |
| 1000 | 14.209 | 5.521 | 181.124 | 19.730 | 0.121x | 0.431x |
| 5000 | 70.834 | 27.867 | 35.885 | 98.701 | 0.609x | 2.157x |
| 10000 | 145.000 | 55.563 | 17.998 | 200.563 | 1.214x | 4.382x |
| 25000 | 390.185 | 139.245 | 7.182 | 529.430 | 3.043x | 11.568x |

## Optimization Findings

- **low_mfu_short_request_overhead**: The H100s are not queue-bound; short one-token scoring requests leave most peak FLOPs unused. Fix: Use compact scoring prompts, then sweep concurrency/max-num-batched-tokens upward after the quality gate passes. Status: compact benchmark prompt added; production prompt should only change after measured quality is present.
- **threshold_calibration**: The model separates positives, but the default 0.5 threshold throws away recall. Fix: Use the recommended threshold for the demo operating point, then validate on a larger gold set. Status: applied
- **prompt_prefill_cost**: Most work is prompt prefill for a one-token classifier. Fix: Default future benchmark runs to a compact prompt variant and report prompt tokens per request. Status: applied in Modal OpenAI-server benchmark path.

## Caveats

- RAG backend is the repo's pure-Python hashing-vectorizer fallback, not tuned FAISS or a neural embedding model.
- RAG numbers are retrieve-only: no LLM answer generation, no cross-encoder rerank, and no semantic continuity across refine turns.
- The 6xH100 reference is the vLLM OpenAI-server semantic scoring benchmark with max_tokens=1 and MFU metrics enabled.
- Fresh-file total models the structural RAG requirement to rebuild embeddings/index before the new document is retrievable.

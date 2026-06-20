# Phase 04 RAG vs 6xH100 Baseline

- commit: `2ae036f`
- query: `GPU queue saturation and throughput metrics`
- RAG backend: `numpy-fallback`
- 6xH100 run: `modal-openai-server-1781943051`
- 6xH100 throughput: `1578.154` req/s, `91286.318` tok/s
- 6xH100 latency: p50 mean `45.765` ms, p95 max `182.296` ms
- 6xH100 derived MFU: `0.060593` of H100 BF16 peak

| docs | RAG index ms | RAG retrieve p50 ms | RAG retrieve qps | fresh-file total ms | retrieve/vLLM p50 | fresh/vLLM p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.108 | 0.046 | 21778.428 | 0.154 | 0.001x | 0.003x |
| 100 | 1.372 | 0.546 | 1830.104 | 1.918 | 0.012x | 0.042x |
| 1000 | 13.942 | 5.486 | 182.272 | 19.428 | 0.120x | 0.425x |
| 5000 | 69.695 | 27.479 | 36.391 | 97.175 | 0.600x | 2.123x |
| 10000 | 140.842 | 55.126 | 18.140 | 195.968 | 1.205x | 4.282x |
| 25000 | 377.285 | 138.312 | 7.230 | 515.596 | 3.022x | 11.266x |

## Caveats

- RAG backend is the repo's pure-Python hashing-vectorizer fallback, not tuned FAISS or a neural embedding model.
- RAG numbers are retrieve-only: no LLM answer generation, no cross-encoder rerank, and no semantic continuity across refine turns.
- The 6xH100 reference is the vLLM OpenAI-server semantic scoring benchmark with max_tokens=1 and MFU metrics enabled.
- Fresh-file total models the structural RAG requirement to rebuild embeddings/index before the new document is retrievable.

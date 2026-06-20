# Infinite-Compute 3-Axis Framework

## The single bottleneck in current RAG

Every design choice in RAG — storing embeddings, building ANN indexes, retrieving a fixed top-k,
generating once — is a response to one constraint: **inference is expensive**. The system spends
memory and storage to avoid inference. As inference cost approaches zero, those tradeoffs invert.

This framework describes the three places where RAG makes an "avoid inference" tradeoff, and how
our architecture removes each bottleneck as compute scales.

---

## Axis 1 — Memory: what you store

**RAG's bottleneck.** Pre-compute embeddings and build an index to avoid re-inferring similarity
at query time. Memory consumed scales as O(N × embedding_dim). It grows with corpus size and does
not shrink with faster compute — the index just sits there. More compute makes retrieval faster
but does not let you access more of the corpus or hold a larger effective context.

**Our design.** Store only raw text. Recompute relevance scores at query time via single-token
scoring. No embeddings, no index. As compute scales, we score more chunks per query — the corpus
in scope grows directly with compute budget.

**Scaling property.** The scoring primitive is embarrassingly parallel: each chunk is scored
independently. Two times the compute (via data-parallel replicas) means two times the throughput,
which means either two times the corpus scored at the same latency, or the same corpus in half the
time. Memory capacity scales **linearly** with compute.

**Current architecture.** All N chunks are scored at query time against the active predicate.
No embedding store. No index.

**Should implement.** Explicit compute-budget parameter that directly controls corpus size scored
per query, making the linear scaling relationship an explicit knob.

---

## Axis 2 — Movement: what you move

**RAG's bottleneck.** The decision of *what to move* to the downstream LLM is made **before any
semantic understanding**, using a cheap proxy (embedding cosine similarity), and **per chunk
independently** (fixed top-k). Because each chunk is judged in isolation, the top-k can be ten
chunks all making the same point — you move ten chunks' worth of bytes and deliver one chunk's
worth of information, while silently missing the parts of the query those chunks don't address.
Faster compute makes retrieval quicker but does not make this decision smarter: top-k is still
top-k, and improving recall means *raising* k, moving *more* bytes.

**Our design.** We invert the order: do the expensive scoring first, then decide what to move
based on exact semantic signal. The movement decision sits **downstream** of the compute, so more
compute buys a smarter decision. This runs in two modes on the same cached scores:

### Mode A — Threshold (cheap, default)
Move every chunk whose calibrated P(Yes) clears a cutoff. The cutoff can be auto-set to a target
precision (≈ mean P(Yes) of the selected set ≥ 0.85) rather than dragged manually. Per-chunk,
independent, no extra inference — strictly better than fixed top-k because the signal is exact, not
a proxy.

### Mode B — Smart selection (spends extra compute)
Choose the *set* that jointly maximises coverage and precision, not just the highest individual
scorers. Decompose the query into facets, score each chunk against each facet (these scores are
cached), then search for the subset that covers the most facets within a movement budget K — a
maximum-coverage objective. This deliberately moves a lower-scoring chunk when it's the only one
covering a facet, and drops a redundant high-scorer. The result is a **smaller, more informative**
set reaching the LLM.

The search is over output subsets — exponential (`C(N, K)`) — but because the objective is
evaluated from **cached per-chunk-per-facet scores**, the search touches no model: scoring is paid
once (linear, parallel), and the combinatorial part is pure arithmetic. Beam width B is the dial:

| compute | method | guarantee |
|---|---|---|
| low | threshold (Mode A) | exact per-chunk signal |
| more | greedy submodular selection | (1 − 1/e) ≈ 63% of optimal |
| more still | beam search, width B | interpolates greedy → exhaustive |
| infinite | exhaustive subset search | exact optimum |

**Scaling property.** Smart selection scales **with compute**: wider beam → closer to the optimal
moved set → fewer, better bytes downstream, with a guaranteed greedy floor and a monotone climb to
the optimum. This is the **same beam-search mechanism as Axis 3, but a separate search** — Axis 3's
beam runs at the input end over *predicates* (producing the survivor set), Axis 2's runs at the
output end over *subsets to move*. They draw from **one global `compute_budget`**, split between
them; the allocation is itself workload-dependent (noisy query → spend on Axis 3; clean query with
redundant survivors → spend on Axis 2).

**Current architecture.** Scores are cached; the user drags a histogram threshold (Mode A,
manual).

**Should implement.** Auto-threshold to a precision target (completes Mode A). Facet decomposition
+ cached facet scores + beam search over output subsets (Mode B), gated by `compute_budget` so the
smart-selection mode turns on and widens as compute scales.

---

## Axis 3 — Truth: answer accuracy

**RAG's bottleneck.** Retrieve once, generate once. If the retrieved context is wrong or
incomplete, the answer is wrong with no self-correction path. Faster compute makes the one-shot
answer arrive sooner but does not improve its accuracy — there is no mechanism to trade compute
for better results in RAG.

**Our design.** Iterate over predicate refinements until the evidence set converges on a correct
answer. The key question is who or what drives the iteration. We support three modes on the same
backend, toggled by a `beam_width` parameter (or compute budget):

### Mode 1 — Human (beam_width = 1, driven externally)
The human inspects the scored evidence through the UI, decides what clause to add or remove, and
judges when the results are good enough. The human is the evaluator. Requires zero additional
compute beyond scoring.

### Mode 2 — Agent via MCP (beam_width = 1, driven externally)
An AI agent calls the same endpoints through an MCP server. The agent inspects scored results,
decides which clause to try next, and judges when to stop — doing exactly what the human does in
mode 1 but programmatically. The agent is the evaluator. The backend is identical; only the
caller changes.

### Mode 3 — Infinite compute (beam_width = N, driven internally)
With abundant compute, try all candidate clause combinations in parallel rather than sequentially.
For each combination, score the corpus and produce a candidate evidence set. An objective function
selects the best combination — for example, highest mean P(Yes) of selected chunks at a minimum
coverage threshold. No human or agent evaluator is needed; the objective function is the
evaluator.

`beam_width` is the compute dial:
- `beam_width = 1`: one clause tried per turn — agent or human decides which one (modes 1 and 2)
- `beam_width = N`: N clause combinations tried in parallel per turn — objective function selects
- As compute scales, widen the beam until the search approaches exhaustive

**Scaling property.** In mode 3, the quality of the selected predicate combination improves
monotonically with beam width. Doubling compute doubles the number of combinations explored.
Truth quality scales **linearly with compute** until the search saturates the clause vocabulary.

**Current architecture.** Manual refine loop (mode 1). Agent loop in Extension 03 approximates
mode 2 with a deterministic candidate expansion policy.

**Should implement.** MCP server exposing the existing `/query`, `/refine`, and `/results`
endpoints as tools (enables mode 2). Beam search over a generated clause vocabulary with an
objective function selector (enables mode 3). The `beam_width` parameter is the toggle between
modes 2 and 3.

---

## Summary

| Axis | RAG bottleneck | Our design | Scales with compute? |
|---|---|---|---|
| Memory | Index size fixed; more compute doesn't expand scope | Score on demand; corpus in scope grows with compute | Yes — linearly |
| Movement | Move-decision made on a cheap proxy, per-chunk (fixed top-k) | Score first, then choose: threshold → beam search over output subsets | Yes — beam width scales the selection toward optimal |
| Truth | One-shot; no self-correction | Three modes on one backend: human → agent → beam search | Yes — beam width scales linearly with compute |

**Rule of thumb.** Use extra compute to score more of the corpus (memory), make a smarter decision
about what to move downstream (movement), and widen the beam over predicate combinations (truth) —
all drawing from one global compute budget.

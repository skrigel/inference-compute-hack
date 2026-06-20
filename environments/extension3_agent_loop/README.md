# extension3-agent-loop

Synthetic dynamic-corpus query-refinement environment for the Extension 3
Applied AI experiment.

The model sees an underspecified search query, positive evidence passages, hard
negative passages, and background distractors. It must output a refined query,
selected evidence ids, exclusion terms, and a stop decision. The reward gives
credit for target-term coverage, positive evidence recall, hard-negative
rejection, query improvement, and avoiding broad "select everything" behavior.

## Quickstart

```bash
prime eval run inference/extension3-agent-loop -n 6 -r 2
```

Environment args:

| arg | default | meaning |
|---|---:|---|
| `split` | `train` | `train` or `eval` |
| `max_examples` | `-1` | limit dataset rows |
| `include_hard` | `true` | include sparse/hard-distractor examples |
| `passages_per_prompt` | `16` | candidate passages included in each prompt |

Primary metrics:

- `target_term_coverage`
- `evidence_id_recall`
- `hard_negative_rejection`
- `exclude_term_use`
- `initial_query_gain`
- `anti_select_all`
- `reward`

Default row counts:

- train: 1,152 rows across 12 retrieval domains
- eval: 288 rows across the same domains with disjoint task ids

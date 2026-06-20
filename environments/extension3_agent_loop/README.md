# extension3-agent-loop

Synthetic dynamic-corpus query-refinement environment for the Extension 3
Applied AI experiment.

The model sees an initial underspecified search query, positive evidence hints,
and near-miss distractors. It must output a refined query. The reward gives
credit for covering target evidence terms, improving over the initial query, and
not producing an overly broad "select everything" query.

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

Primary metrics:

- `target_term_coverage`
- `initial_query_gain`
- `anti_select_all`
- `reward`

# Agent Experiment Summary Format

After completing an experiment, the agent MUST generate a summary in this exact format.
This summary serves as the canonical record for the experiment and should be saved to
`eval/artifacts/experiment_summaries/OPT-XXX_summary.md`.

---

```markdown
# Experiment Summary: OPT-XXX - [Short Name]

Generated: [ISO timestamp]
Agent: [agent identifier]
Commit: [git commit hash]

## Executive Summary

[2-3 sentence summary of what was tested and the key finding]

**Verdict:** [ACCEPTED | REJECTED | INCONCLUSIVE]
**Confidence:** [HIGH | MEDIUM | LOW]

---

## 1. Experiment Design

### 1.1 Hypothesis
[Clear statement of what improvement was expected and why]

### 1.2 Independent Variables
| variable | values tested |
|---|---|
| [e.g., batch_size] | [e.g., 128, 256, 512, 1024] |

### 1.3 Controlled Variables
| variable | held constant at |
|---|---|
| [e.g., model] | [e.g., Qwen/Qwen2.5-3B-Instruct-AWQ] |

### 1.4 Dataset Configuration
| size tier | doc count | corpus description |
|---|---:|---|
| small | | |
| medium | | |
| large | | |
| xlarge | | |
| xxlarge | | |

### 1.5 Repetition Protocol
- Total repetitions per configuration: [n]
- Warmup runs excluded: [yes/no, count]
- Outlier exclusion criterion: [e.g., >3σ, none]
- Effective sample size after exclusions: [n]

---

## 2. Raw Results

### 2.1 Per-Run Data

<details>
<summary>Expand raw run data</summary>

| run_id | dataset_size | config | rep | metric_1 | metric_2 | ... | timestamp |
|---|---|---|---:|---:|---:|---|---|
| | | | | | | | |

</details>

### 2.2 Aggregated Results by Configuration

| config | dataset | n | metric mean | metric std | metric min | metric max | 95% CI |
|---|---|---:|---:|---:|---:|---:|---|

---

## 3. Statistical Analysis

### 3.1 Normality Tests
| config | dataset | metric | Shapiro-Wilk W | p-value | normal? |
|---|---|---|---:|---:|---|

### 3.2 Variance Homogeneity
| comparison | Levene F | p-value | homogeneous? |
|---|---:|---:|---|

### 3.3 Hypothesis Tests
| comparison | test used | test statistic | p-value | effect size | significant? |
|---|---|---:|---:|---:|---|

### 3.4 Confidence Intervals
| metric | baseline 95% CI | candidate 95% CI | overlap? |
|---|---|---|---|

---

## 4. Scaling Analysis

### 4.1 Dataset Size Scaling
| metric | small | medium | large | xlarge | xxlarge |
|---|---:|---:|---:|---:|---:|
| [e.g., req/s] | | | | | |
| [e.g., p50 ms] | | | | | |

### 4.2 Scaling Factors
| transition | throughput factor | latency factor | complexity class |
|---|---:|---:|---|
| small → medium | | | [O(1), O(n), O(n²), etc.] |
| medium → large | | | |
| large → xlarge | | | |
| xlarge → xxlarge | | | |

### 4.3 Scaling Regression
| metric | regression model | R² | equation | extrapolation warning |
|---|---|---:|---|---|

---

## 5. Quality Gate Validation

### 5.1 Per-Dataset Quality
| dataset | precision mean ± std | recall mean ± std | F1 mean ± std | threshold | pass? |
|---|---:|---:|---:|---:|---|

### 5.2 Quality Degradation Check
| comparison | baseline F1 | candidate F1 | delta | within tolerance? |
|---|---:|---:|---:|---|

---

## 6. Resource Utilization

### 6.1 GPU Metrics
| config | dataset | GPU util mean ± std | GPU util max | MFU mean ± std |
|---|---|---:|---:|---:|

### 6.2 Memory Metrics
| config | dataset | VRAM used MB | VRAM max MB | OOM events |
|---|---|---:|---:|---:|

### 6.3 Power Metrics
| config | dataset | power mean ± std W | power max W | energy per request mJ |
|---|---|---:|---:|---:|

---

## 7. Comparison to Baseline

### 7.1 Performance Delta
| workload | dataset | baseline | candidate | abs delta | rel delta % | p-value | verdict |
|---|---|---:|---:|---:|---:|---:|---|

### 7.2 Regression Check
| metric | regression threshold | observed delta | pass? |
|---|---:|---:|---|

---

## 8. Anomalies and Observations

### 8.1 Outliers Detected
| run_id | metric | value | z-score | action taken |
|---|---|---:|---:|---|

### 8.2 Unexpected Behaviors
[List any unexpected patterns, errors, or anomalies observed during the experiment]

### 8.3 Infrastructure Issues
[Note any infrastructure problems: OOMs, timeouts, network issues, etc.]

---

## 9. Conclusions

### 9.1 Hypothesis Evaluation
- **Supported:** [yes/no/partially]
- **Evidence strength:** [strong/moderate/weak]
- **Confounding factors:** [list any]

### 9.2 Recommendation
[Clear recommendation: apply, reject, or retest with modifications]

### 9.3 Limitations
[Known limitations of this experiment]

### 9.4 Follow-up Experiments
[Suggested next experiments based on findings]

---

## 10. Artifacts

| artifact | path | checksum |
|---|---|---|
| raw results JSON | | |
| aggregated results | | |
| plots | | |
| Weave upload receipt | | |

---

## 11. Reproducibility

### 11.1 Environment
```
python: [version]
vllm: [version]
torch: [version]
cuda: [version]
gpu: [model]
```

### 11.2 Command to Reproduce
```bash
[exact command to reproduce this experiment]
```

### 11.3 Random Seeds
| component | seed |
|---|---|
```

---

## Usage Instructions for Agents

### Complete Workflow

1. **Create experiment folder structure** (see folder layout below)
2. **Save config.json** before running experiments
3. **Save individual run results** as `run_NNN.json` after each run
4. **Compute aggregated statistics** and save `aggregated.json`
5. **Generate this summary** and save to `eval/artifacts/experiment_summaries/OPT-XXX_summary.md`
6. **Append entry to ledger** in `docs/optimization-results-ledger.md`

### Folder Structure

```
eval/artifacts/
├── experiment_summaries/
│   └── OPT-XXX_summary.md          # This summary document
├── experiment_results/
│   └── OPT-XXX/
│       ├── config.json             # Experiment configuration
│       ├── runs/
│       │   ├── run_001.json        # Individual run results
│       │   ├── run_002.json
│       │   └── ...
│       ├── aggregated.json         # Aggregated statistics
│       ├── scaling_analysis.json   # Dataset scaling data
│       └── plots/
│           └── ...
```

### Summary Requirements

1. **All sections are required** - use "N/A" or "Not measured" only if truly inapplicable
2. **Raw data must be preserved** - the collapsible raw data section is mandatory
3. **Statistical tests are mandatory** - do not claim significance without p-values
4. **Link to the optimization ledger** - ensure the corresponding OPT-XXX entry exists in `docs/optimization-results-ledger.md`
5. **Reference detailed results** - Section 10 (Artifacts) must point to the `experiment_results/OPT-XXX/` folder

## Minimum Viable Summary

If time-constrained, the following sections are **absolutely required**:
- Executive Summary with verdict
- Section 1.4 (Dataset Configuration)
- Section 1.5 (Repetition Protocol)
- Section 2.2 (Aggregated Results)
- Section 3.3 (Hypothesis Tests)
- Section 4.2 (Scaling Factors)
- Section 7.1 (Performance Delta)
- Section 9.2 (Recommendation)
- Section 11.2 (Command to Reproduce)

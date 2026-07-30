# SaliTrap: Revealing the Salience Bias of Large Language Models in Commonsense Reasoning

![SaliTrap Pipeline](https://github.com/Wuzheng02/SaliTrap/blob/main/pipeline.png)


Large language models (LLMs) have learned to heavily prioritize explicit conditions provided in the input. We term the resulting vulnerability **Salience Bias**: models become easily hijacked by useless explicit distractors (e.g., numerical values), leading them to ignore the implicit physical or commonsense prerequisites of a task. To study this phenomenon, we construct **SaliTrap**, a high-quality benchmark spanning four trap dimensions (missing prerequisite, environmental mismatch, temporal/physiological violation, and rule mismatch), and show that this failure mode is overwhelmingly a matter of **knowledge suppression rather than knowledge absence**.

This repository provides:

- The **data synthesis pipeline** used to construct SaliTrap (seed scaling, candidate generation/validation, iterative refinement, and stability retesting).
- The **evaluation pipeline** used to benchmark LLMs on SaliTrap and reproduce the paper's tables.


## Dataset

The SaliTrap benchmark (1,145 items across four trap dimensions) is released on Hugging Face:

**Dataset**: [https://huggingface.co/datasets/TBD/SaliTrap](https://huggingface.co/datasets/TBD/SaliTrap) *(link will be updated upon release)*


## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install requests tqdm
```

The codebase only depends on the Python standard library plus `requests` (required) and `tqdm` (optional, used for progress bars; a minimal fallback is provided if it is not installed).

Tested with Python 3.9+.

## Quick Start

All scripts talk to an OpenAI-compatible chat-completion gateway. Set your credentials once via environment variables:

```bash
export PHYSTRAP_APP_KEY="your_api_key"
export PHYSTRAP_BASE_URL="https://your-gateway.example.com/v1/chat/completions"
```

### 1. Seed Scaling

Expand the hand-written prototype seeds into a larger pool of commonsense-trap seeds:

```bash
python seed_scaler/phystrap_seed_scaler.py --app-key $PHYSTRAP_APP_KEY \
    --per-dimension 5 \
    --export scaled_seeds.py
```

This produces `logs/PhysTrap_SeedScaler_{timestamp}.json`, a seed list directly consumable by the synthesis pipeline below.

### 2. Data Synthesis Pipeline

Run the full synthesis pipeline (seed -> candidate generation -> tri-checker validation -> Solver-Judge -> iterative refinement):

```bash
python run_pipeline.py --app-key $PHYSTRAP_APP_KEY \
    --seed-file ./seeds/scaling_final.json \
    --output-dir ./outputs \
    --workers 20
```

Key options:

| Option | Description | Default |
|---|---|---|
| `--seed-file` | Path to the input seed JSON file | `./seeds/scaling_final.json` |
| `--output-dir` | Output directory for synthesis results | `./outputs` |
| `--workers` | Number of parallel worker threads | `20` |
| `--generator-rounds` | Max rounds for the Generator | `2` |
| `--anneal-rounds` | Max rounds for annealing | `3` |
| `--max-candidates` | Per-seed candidate cap | `12` |
| `--resume` | Resume from checkpoint | `True` |

The pipeline is checkpointed: interrupted runs automatically skip already-completed seeds on restart, and progress is shown via a live `tqdm` bar.

### 3. Stability Retest

Once candidates have been accepted by the main pipeline, re-verify their stability under repeated sampling:

```bash
python stability_retest/stability_retest.py \
    --cases stability_retest/candidate_cases.json \
    --repeats 5
```

An item is labeled `strong_stable` if `strong_attack_rate >= 0.6`, and `soft_stable` if `attack_rate >= 0.8`.

### 4. Evaluation

Benchmark target LLMs on the resulting `results_full.json`:

```bash
python eval/eval_phystrap.py --app-key $PHYSTRAP_APP_KEY \
    --workers 50 --resume
```

Each target model acts as the Solver; a single fixed Judge model (kept constant across all target models for a fair comparison) assigns a failure-mode label (`Strict Pass`, `Sycophantic Compliance`, `CoT Hijacked`, `Hard Fail`, `Patch Compliance`, `Refusal`). Per-model results are written to `eval/eval_results/*.jsonl`, and Trap-Attack-Rate (TAR) / Hard-Fail-Rate (HFR) are aggregated per dimension and overall.

Track progress with:

```bash
wc -l eval/eval_results/*.jsonl
```

### 5. Generating LaTeX Tables

Reproduce the paper's result tables from the evaluation output:

```bash
# SCR + Sycophancy Index (SI)
python eval/gen_latex_table_sycophancy.py

# SCR + Conditional Sycophancy Index (Cond-SI)
python eval/gen_latex_table_sycophancy_condsi.py
```

## Configuration

All scripts are configured via environment variables (with sensible defaults / CLI overrides):

| Variable | Description | Default |
|---|---|---|
| `PHYSTRAP_APP_KEY` | Bearer token / API key for your LLM gateway | — |
| `PHYSTRAP_BASE_URL` | Chat-completion endpoint (OpenAI-compatible) | `https://api.example.com/v1/chat/completions` |
| `PHYSTRAP_MODEL_POOL` | Comma-separated model pool used for 429-rate-limit rotation | 5-model example pool |
| `PHYSTRAP_TEMPERATURE_FIXED_1` | Comma-separated model names that only accept `temperature=1` | empty |
| `PHYSTRAP_DATASET` | Path to `results_full.json` produced by the synthesis pipeline | `outputs/results_full.json` |
| `PHYSTRAP_OUTPUT_DIR` | Output directory for per-model evaluation `.jsonl` files | `eval/eval_results` |
| `PHYSTRAP_JUDGE_MODEL` | Fixed judge model id used for all target models | `judge-model` |
| `PHYSTRAP_EVAL_RESULTS_DIR` | Directory containing per-model result files (for table generation) | `eval/eval_results` |
| `PHYSTRAP_EVAL_TOTAL` | Expected total item count, used as a sanity check | `1145` |

Before running any script against your own infrastructure, replace the placeholder model names in `api_client.py` (`DEFAULT_MODEL_POOL`) and `eval/eval_phystrap.py` (`EVAL_MODELS`) with the actual model ids/gateway names you want to use.

## Citation

If you find this work useful, please cite:

```bibtex

```

# AAI3008 Large Language Models - Group 11

This branch focuses on TinyStories experiments comparing:

1. an optimized GPT-style baseline model (91.7M params), and
2. a fine-tuned EleutherAI Pythia-70M model.

The repository currently stores notebook pipelines plus exported evaluation artifacts (metrics, generated stories, and OOV analysis).

## Branch Snapshot

Current branch files:

```
.
|- TinyStories_SLM.ipynb
|- Pythia_TinyStories.ipynb
|- baseline_91m_metrics.json
|- baseline_91m_stories.json
|- pythia_70m_metrics.json
|- pythia_70m_stories.json
|- oov_comparison.json
`- README.md
```

## What Each Notebook Does

### `TinyStories_SLM.ipynb`
- Installs dependencies and configures a Colab-style environment.
- Loads TinyStories and runs ablation-style setup for tokenizer/corpus experiments.
- Trains and evaluates the custom GPT-style baseline pipeline.
- Produces baseline outputs used in this branch artifacts.

### `Pythia_TinyStories.ipynb`
- Loads `EleutherAI/pythia-70m` tokenizer/model.
- Tokenizes TinyStories with max sequence length 512.
- Runs gentle fine-tuning (1 epoch) with scheduler and early stopping.
- Exports generated stories and evaluation metrics for model comparison.

## Exported Artifact Files

- `baseline_91m_metrics.json`: aggregate and per-prompt judge scores for the optimized baseline.
- `baseline_91m_stories.json`: generated completions for baseline evaluation prompts.
- `pythia_70m_metrics.json`: aggregate and per-prompt judge scores for fine-tuned Pythia-70M.
- `pythia_70m_stories.json`: generated completions for Pythia-70M evaluation prompts.
- `oov_comparison.json`: out-of-vocabulary token rate comparison across both models.

## Headline Results (From JSON Artifacts)

Aggregate scores (higher is better):

| Model | Grammar | Creativity | Consistency | Plot | Overall |
|---|---:|---:|---:|---:|---:|
| Optimized Baseline 91.7M | 8.09 | 6.10 | 8.23 | 6.17 | 7.15 |
| Fine-tuned Pythia-70M | 7.97 | 5.78 | 7.99 | 6.00 | 6.94 |

Overall delta (Baseline - Pythia):
- Grammar: +0.12
- Creativity: +0.32
- Consistency: +0.24
- Plot: +0.17
- Overall: +0.21

OOV rate comparison (lower is better):

| Model | OOV Rate (%) |
|---|---:|
| Fine-tuned Pythia-70M | 0.0713775874 |
| Optimized Baseline 91.7M | 0.0682128240 |

## Reproduction Notes

This branch is notebook-first. Run the notebooks in order below if you want to reproduce artifacts:

1. `TinyStories_SLM.ipynb`
2. `Pythia_TinyStories.ipynb`

Typical package set used in notebooks:

```bash
pip install datasets transformers sentencepiece tokenizers accelerate torch matplotlib seaborn wandb
```

Several cells are written for Google Colab (`/content/...`, Drive mount). If you run locally, update the path variables in the setup cells.

## Dataset

- TinyStories: https://huggingface.co/datasets/roneneldan/TinyStories

## Branch-Specific Note

Older references to `PrePhase.ipynb` and `full_pipeline_commented.ipynb` do not apply to this branch. The active workflow here is the pair of notebooks listed above plus the exported JSON artifacts.


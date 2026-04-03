# AAI3008 - Large Language Models: Group 11
## A Dive into Large Language Model Configurations & Optimization Strategies

This repository contains the full experimental pipeline our project. The central question is whether a **smaller, carefully designed model** can match the story-generation quality of much larger off-the-shelf alternatives and what configuration decisions get it there.

---

## Motivation

LLM success is often attributed to scale alone. This project challenges that assumption by systematically isolating the effect of individual design decisions: tokenizer vocabulary size, context window, model depth, and hyperparameters on a controlled task: **children's story generation** using the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) (2.1 million short stories, limited vocabulary, predictable structure).

---

## Repository Structure

- `experiment_log.json` : detailed per-run logs for hyperparameter sweeps
- `gpt_eval_comparison_20B.json`, `gpt_eval_comparison_20B_cleaned.json`, `gpt_eval_comparison_7M_91M.json` — evaluation outputs from the LLM-as-judge pipeline
- `tiny_corpus.txt` : training corpus (preprocessed TinyStories subset) (Due to large file size, will upload it during submission via link)
- `tokenizer.json`, `vocab.json` — custom BPE tokenizer assets
- Jupyter notebooks for experiments and evaluation (e.g., `full_pipeline.ipynb`, `evaluation.ipynb`, `PrePhase.ipynb`)
- Lora folder : Lora implementation

---

## Methodology Overview

The project is divided into four phases:

### Pre-Phase - Do We Even Need Attention?
Built from scratch in `PrePhase.ipynb`:
- **Bigram LM** (character-level): val loss 2.3012 - largely incoherent output
- **MLP LM** (character-level): val loss 1.1709 - readable words, no coherence
- **Causal Attention model**: val loss 1.0752 - coherent sentences, clear winner

**Conclusion:** Attention is necessary. A GPT-2 style model is adopted as the baseline going forward.

---

### Phase 1 - Component Studies (what makes a well-configured model?)

All studies hold compute constant (1 epoch, same hardware) and vary one factor at a time against a 7M parameter GPT-2 style baseline.

#### Tokenizer & Corpus Size
- Trained a custom **Byte-Level BPE tokenizer** on the TinyStories corpus.
- Swept vocabulary sizes from 2,048 → 50,257 tokens.
- Average tokens per story drops sharply from 250 → 220 between 2K and 8K vocab (12% compression gain), then plateaus.
- **Sweet spot: 8,000 tokens** - best compression-to-size ratio, max token count stabilises (cutting truncation risk).
- 8-gram containment on 10K stories is only 7.9%, confirming the model learns narrative patterns rather than memorising training data.

#### Context Window
- Swept context lengths: 128, 256, 384, 512, 640, 768, 1020, 2048.
- Larger windows → fewer training blocks per epoch → fewer optimiser steps → worse update efficiency.
- **Sweet spot: 512 tokens** - lowest validation loss (1.644), ~50 minutes per epoch.

#### Model Size (Layers & Embedding Dimension)

| Config | Parameters | Val Loss | Training Time |
|--------|-----------|----------|---------------|
| 6L · 8H · D=256 | 7M | 1.645 | ~45 min |
| 8L · 8H · D=384 | 17.5M | 1.469 | ~120 min |
| 12L · 12H · D=768 | 128M | 1.276 | ~480 min |

Larger models improve loss substantially, but at a steep compute cost.

#### Hyperparameter Studies

**Learning Rate** (AdamW, CE loss, 1 epoch):

| LR | Val Loss | Perplexity |
|----|----------|------------|
| 1e-3 | 1.6395 | 5.153 ✅ |
| 5e-4 | 1.6451 | 5.182 |
| 3e-4 | 1.7774 | 5.914 |
| 2e-1 | 3.0118 | 20.32 |
| 5e-6 | 4.0858 | 59.49 |

→ **Best: LR = 1e-3**

**Optimizer** (LR = 1e-3, CE loss):

| Optimizer | Val Loss | Perplexity |
|-----------|----------|------------|
| Adam | 1.6385 | 5.148 ✅ |
| AdamW | 1.6395 | 5.153 |
| Lion | 1.6404 | 5.157 |
| SGD | 6.2060 | 495.71 ❌ |

→ **Best: Adam** (SGD fails entirely - non-adaptive optimisers are unsuitable for transformers).

**Label Smoothing** (Adam, LR = 1e-3):

| ε | Val Loss | Perplexity |
|---|----------|------------|
| 0 | 1.6385 | 5.148 ✅ |
| 0.001 | 1.6591 | 5.254 |
| 0.1 | 2.7516 | 15.66 |
| 0.2 | 3.6737 | 39.40 |

→ **Best: No label smoothing.** TinyStories has highly deterministic token sequences, making overconfidence a non-issue and smoothing an inappropriate regulariser for this domain.

**Final configuration selected:** LR = 1e-3, Adam, CE loss (no smoothing).

---

### Phase 2 - Model Comparisons (1 epoch each)

**Custom model configuration (91.6M parameters):**
```python
GPT2Config(
    vocab_size=8000,
    n_positions=512,
    n_ctx=512,
    n_embd=768,
    n_layer=12,
    n_head=12
)
# LR = 1e-3, Adam, CE loss
```

| Comparison | Grammar | Creativity | Consistency | Plot | Notes |
|------------|---------|------------|-------------|------|-------|
| 91.6M Custom vs 7M Baseline | **Custom wins** | **Custom wins** | **Custom wins** | 7M wins | - |
| 91.6M Custom vs GPT-Neo 33M (multi-epoch, open-source) | Neo wins | Neo wins | Neo wins | Neo wins | Unfair - Neo trained for many more epochs |
| 91.6M Custom vs GPT-Neo 33M (1 epoch, same setting) | **Custom wins**| **Custom wins** | **Custom wins** | Neo wins | Fair comparison, custom model holds its own |
| 91.6M Custom vs GPT-NeoX 20B (open-source) | 20B wins | 20B wins | 20B wins | **Custom wins** | Before stripping mid-generation story restarts. Custom model wins Plot only |
| 91.6M Custom vs GPT-NeoX 20B (open-source) | 20B wins | 20B wins | **Custom wins** | **Custom wins** | After stripping mid-generation story restarts. Custom model  win Consistency + Plot and leads A/B test 23–17 |

**Key insight:** Training a domain-specific model on a task-specific dataset can outperform a general model 200× its size on the target task.

---

### Phase 3 - Fine-Tuning vs Training from Scratch

Full fine-tuning and LoRA experiments comparing the custom model against **pythia-70M**.

---

## Evaluation Methodology

Stories are judged using an **LLM-as-a-judge** framework, adapted from the TinyStories paper:

- GPT-4 / LLaMA 3.1 8B is prompted to act as a teacher grading a student's story completion.
- 50 story seeds × 10 completions = 500 evaluated stories per model.
- Four metrics scored 1–10: **Grammar, Creativity, Consistency, Plot**.
- GPT-4's scores correlate with human judgement at Pearson r ≈ 0.882, often exceeding inter-human agreement.

**Known limitations of LLM-as-a-judge:**
1. **Narcissistic bias** - judge may favour outputs resembling its own family (e.g., GPT-4 rating GPT-2 outputs higher).
2. **Verbosity bias** - longer stories are rated higher regardless of quality.
3. **Domain limits** - TinyStories models fail general benchmarks (BLiMP, EWoK) by design; their vocabulary is intentionally restricted to ~1,500 child-level words.

---

## Quick Start

### Environment Setup
```bash
pip install datasets transformers sentencepiece tokenizers accelerate torch
```

### Running the Pre-Phase Experiments
Open and run `PrePhase.ipynb`. It loads TinyStories, builds a character-level vocabulary, and trains Bigram → MLP → Causal Attention models sequentially.

### Running the Full Pipeline
Open `full_pipeline_commented.ipynb`. It is structured as follows:

1. **Data preparation** - download TinyStories, export corpus, train BPE tokenizer (vocab_size = 8000), tokenize and pack into fixed-length blocks.
2. **Hyperparameter studies** - sweep learning rates, optimisers, and label smoothing values.
3. **Baseline training** - 7M GPT-2 style model.
4. **Custom model training** - 91.6M GPT-2 style model with optimal config.
5. **Replicated GPT-Neo 33M** - trained under the same 1-epoch setting for a fair comparison.
6. **Evaluation** - LLM-as-a-judge across all model pairs; A/B human testing for the 91.6M vs 20B comparison.

---

## Key Findings Summary

| Design Decision | Optimal Choice | Reason |
|----------------|---------------|--------|
| Tokenizer vocab size | **8,000** | Best compression-to-size ratio; gains plateau beyond this |
| Context window | **512 tokens** | Balances update efficiency and narrative coverage |
| Model size | **Larger is better** (up to compute budget) | 91.6M >> 7M in val loss |
| Learning rate | **1e-3** | Fastest convergence within 1 epoch |
| Optimizer | **Adam** | Adaptive gradient scaling essential; SGD fails |
| Label smoothing | **None (ε = 0)** | TinyStories is deterministic; smoothing hurts |
| Attention | **Required** | Bigram/MLP models are insufficient for coherent story generation |

**Overall conclusion:** A 91.6M model trained from scratch on a task-specific dataset for a single epoch is competitive with a general-purpose 20B model on the same task, and outperforms it on narrative consistency and plot when evaluated fairly.

---

## Dataset

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) - Eldan & Li, 2023. 2.1 million short English stories generated by GPT-3.5/4, designed to train and evaluate small language models.

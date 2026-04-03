# AAI3008 - Large Language Models: Group 11 - Tien Loc

## A Dive into Large Language Model Configurations — The Effect of Model Size (Layers + Embedding Size) and The Issue with EOS Token

---

## Table of Contents

- [Project Overview](#project-overview)
- [Model Size Experiments](#model-size-experiments)
  - [Base Model](#base-model)
  - [Setting 1](#setting-1)
  - [Setting 2](#setting-2)
  - [Summary](#summary)
- [The Issue with `.Once` — EOS Token Problem](#the-issue-with-once--eos-token-problem)
  - [Root Cause](#root-cause)
  - [Two Observed Issues](#two-observed-issues)
  - [Solution](#solution)

---

## Project Overview

This project explores how model architecture choices — specifically the number of layers and embedding dimensions — affect language model performance on the TinyStories corpus. It also documents a critical training artefact involving the `.Once` token and how it was resolved by introducing a dedicated end-of-story token.

All models are GPT-style autoregressive transformers trained on the TinyStories dataset.

---

## Model Size Experiments

### Base Model

| Parameter | Value |
|---|---|
| `vocab_size` | 8000 (from the tokenizer) |
| `n_positions` | 512 |
| `n_ctx` | 512 |
| `n_embd` | 256 (embedding dimension) |
| `n_layer` | 6 (number of layers) |
| `n_head` | 8 (number of attention heads) |
| **Model size** | **7M** |
| **Val loss** | **1.645133** |
| **Training time** | ~45 minutes / epoch |

---

### Setting 1

Increased embedding dimension from 256 → 384 and layers from 6 → 8.

| Parameter | Value |
|---|---|
| `vocab_size` | 8000 (from the tokenizer) |
| `n_positions` | 512 |
| `n_ctx` | 512 |
| `n_embd` | 384 (embedding dimension) |
| `n_layer` | 8 (number of layers) |
| `n_head` | 8 (number of attention heads) |
| **Model size** | **17.5M** |
| **Val loss** | **1.468510** |
| **Training time** | ~120 minutes / epoch |

---

### Setting 2

Further scaled up: embedding dimension 384 → 768, layers 8 → 12, context window 512 → 1024.

| Parameter | Value |
|---|---|
| `vocab_size` | 8000 (from the tokenizer) |
| `n_positions` | 512 |
| `n_ctx` | 1024 |
| `n_embd` | 768 (embedding dimension) |
| `n_layer` | 12 (number of layers) |
| `n_head` | 12 (number of attention heads) |
| **Model size** | **128M** |
| **Val loss** | **1.276403** |
| **Training time** | ~480 minutes / epoch |

---

### Summary

| Setting | Model Size | Val Loss | Time / Epoch |
|---|---|---|---|
| Base | 7M | 1.645133 | ~45 min |
| Setting 1 | 17.5M | 1.468510 | ~120 min |
| Setting 2 | 128M | 1.276403 | ~480 min |

> **Observation**: Increasing model size (via deeper layers and wider embeddings) consistently improves validation loss, at the cost of significantly longer training time. Going from 7M → 128M parameters reduces val loss by ~0.37.

---

## The Issue with `.Once` — EOS Token Problem

### Root Cause

The TinyStories corpus is heavily dominated by stories that begin with `"Once upon a time…"`. The token frequency breakdown illustrates the imbalance:

| Token | Frequency |
|---|---|
| `Once` | ~1.6M |
| `$once` | ~16K |
| `$Once` | ~4K |

Because this opening phrase appears so frequently in training, the model learned that `.Once` is a strong story-boundary signal — i.e., a period followed immediately by `Once` indicates that a new story should begin.

---

### Two Observed Issues

**Issue 1 — Story continuation mid-output**

~80% of generated completions would hit a `.Once` token mid-output and immediately start generating an entirely new, unrelated story. The LLM judge then scored two concatenated stories as a single coherent completion, penalising custom model scores.

```
"…the bunny hopped home. Once upon a time, there was a dragon who breathed fire…"
                        ^^^^^ new story begins here — unintended
```

**Issue 2 — Missing space in `.Once` token**

Because `.Once` was fused as a single token in the training data (period immediately followed by capital `O`, no whitespace), the model reproduced this exact byte sequence — emitting no space between the sentence-ending period and the word `Once`. This produced malformed text that broke sentence tokenisation downstream.

```
"…she smiled..Once upon a time…"
             ^^ missing space — malformed output
```

---

### Solution

The fix was implemented in three steps at the data and training level — no post-processing heuristics required.

**Step 1 — Add `<|endofstory|>` special token**

A dedicated end-of-story token is appended to every story in the training corpus, giving the model an unambiguous signal that the story is complete — entirely distinct from any naturally occurring word like `Once`.

```
"…and the bunny hopped home happily ever after. <|endofstory|>"
```

**Step 2 — Update chunking to never split `<|endofstory|>` mid-token**

The data pipeline's chunking logic was updated to treat `<|endofstory|>` as an atomic boundary. Chunks always break at story boundaries, ensuring the token is never split across two training sequences.

```
# Before (bad)
Chunk A: "…ever after. <|endof"
Chunk B: "story|> Once upon…"

# After (good)
Chunk A: "…ever after. <|endofstory|>"
Chunk B: "Once upon…"
```

**Step 3 — Train `<|endofstory|>` as the EOS token**

The model is trained to predict `<|endofstory|>` as its end-of-sequence signal. Generation halts cleanly at this token — eliminating spurious `.Once` continuations and concatenated stories entirely.

---

*AAI3008 Large Language Models — Group 11*

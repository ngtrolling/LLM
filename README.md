# LLM — TinyStories experiments

This branch contains experiments for training and evaluating small GPT-style language models on the TinyStories dataset. The goal of the experiments was to identify robust baseline hyperparameters for a small (≈7M) GPT-2-style model trained on short children's stories.

## Key artifacts

- `experiment_log.json` : detailed per-run logs for hyperparameter sweeps
- `gpt_eval_comparison_20B.json`, `gpt_eval_comparison_20B_cleaned.json`, `gpt_eval_comparison_7M_91M.json` — evaluation outputs from the LLM-as-judge pipeline
- `tiny_corpus.txt` : training corpus (preprocessed TinyStories subset)
- `tokenizer.json`, `vocab.json`, `merges.txt` — custom BPE tokenizer assets
- Jupyter notebooks for experiments and evaluation (e.g., `full_pipeline.ipynb`, `evaluation.ipynb`, `base_tiny_stories.ipynb`)
- lora : lora implemenration

---

## Methodology

- Model family: GPT-2-style transformer (custom small variant used for the experiments).
- Typical training config used across experiments: 1 epoch, context length = 512 tokens, vocab size = 8,000 (custom BPE), evaluation on validation perplexity.
- Experiments were run in three phases, varying one hyperparameter group at a time:
  1. Learning rate sweep
  2. Optimizer comparison
  3. Loss function (label smoothing) study

Performance was primarily measured using validation cross-entropy loss and its exponential (perplexity). Lower perplexity indicates the model assigns higher probability to the correct next token on average.

## Hyperparameter studies : summary of findings

### Learning rate study

- Search space included values like: 1e-3, 5e-4, 3e-4, 2e-4, 5e-6, 2e-1, etc.
- Best learning rate: 1e-3 (final validation perplexity ≈ 5.153). This provided sufficient gradient step size to converge within a single epoch on TinyStories.
- Observations:
  - Very low learning rates (e.g., 5e-6) underfit due to insufficient updates.
  - Very high learning rates (e.g., 2e-1) caused training instability and divergence.

Conclusion: For this small model and dataset, 1e-3 balanced stability and speed-of-learning and was selected for later experiments.

### Optimizer study

- Optimizers tested: `adamw` (baseline), `adam`, `lion`, `sgd` (with momentum variants).
- Results:
  - Adaptive optimizers (Adam, AdamW, Lion) achieved near-identical validation perplexities (≈5.14–5.15).
  - `sgd` performed very poorly (perplexity ≈ 495.7) indicating non-adaptive optimizers are unsuitable for training transformers from scratch on this task/scale.
  - `adam` attained the marginally lowest perplexity in the sweep and was selected for Phase 3 (loss-function experiments).

Conclusion: Adaptive gradient mechanisms are crucial at this scale; the exact adaptive variant matters less for TinyStories/7M models.

### Loss function study (label smoothing)

- Setup: Cross-entropy (CE) baseline vs CE with label smoothing (ε = 0.01, 0.1, 0.2 tested).
- Observations:
  - Contrary to common expectations, label smoothing consistently degraded validation performance on TinyStories.
  - Minimal smoothing (ε = 0.01) increased perplexity slightly (e.g., to ≈5.2544). Stronger smoothing (ε = 0.1, 0.2) caused severe degradation (perplexities ~15.66 and ~39.40 respectively).
  - Hypothesis: TinyStories is a deterministic, highly-predictable children's storytelling dataset with a limited vocabulary (~1.5k–8k depending on preprocessing). For this domain, overconfidence is less harmful and label smoothing acts as an inappropriate regulariser.

Conclusion: Do not apply label smoothing by default on TinyStories-like datasets; it is dataset-dependent.

## Final configuration moving forward (recommended baseline)

- Learning rate: 0.001
- Optimizer: `adam`
- Label smoothing: 0.0
- Context length: 512
- Vocab (BPE): 8,000

These hyperparameters represent the best-performing, low-risk configuration for the baseline GPT-2 architecture on the TinyStories data in these experiments.

## Evaluation metric research

- Evaluation protocol: LLM-as-a-judge using a large instruction-following LLM to grade generated stories.
- Setup used in the slides:
  - 50 story seeds selected; each seed completed 10 times by the model → 500 completions.
  - Each completion scored on four metrics: Grammar, Creativity, Consistency, Plot (1–10).
  - The `master prompt` grounds the judge with a clear separator token (`***`) to separate prompt vs generated completion.
  - Temperature used for generation: 1 (for diversity in judged samples).

- Validation of the judge: prior research suggests high-ability LLM judges (e.g., GPT-4) can achieve a Pearson correlation with human judgment ≈ 0.88 on similar tasks. Still, automatic judges have well-known biases (see Limitations).

## Master prompt (judge grounding)

The master prompt used to instruct the judge follows this pattern (paraphrased):

"The following exercise: the student is given the beginning of a story and must complete it. The symbol `***` marks the separator between the prescribed beginning and the completion: [Context]***[prediction]. Please provide a general assessment for the student's completion and then grade the completion on Grammar (1–10), Creativity (1–10), Consistency (1–10), and Plot (1–10)."

This prompt emphasizes the separator to prevent the judge from confusing prompt content with the student completion.

## Limitations and known biases of the evaluation

- Narcissistic bias: LLM judges may prefer outputs that resemble their own training/style.
- Verbosity bias: Judges often reward longer, more detailed completions even if the concise completion is of equal quality.
- Domain limits: TinyStories-trained models are domain-specific and struggle on broad linguistic benchmarks (e.g., BLiMP) due to limited vocabulary and narrow world knowledge.

These caveats should be considered when interpreting absolute metric numbers and when transferring the baseline to very different datasets or larger architectures.

## Lora Implementation

- Lora did not result in favourable results due to the short training run. Increasing the epoch, did drop the validation loss but it was not enough to generate coherent stories. Therefore, next steps would be to adjusting the hyperparameters of Lora which is a critical aspect of its success.

---

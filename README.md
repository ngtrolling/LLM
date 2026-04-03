# LLM - Character-Level Bigram Language Model

This folder contains a minimal PyTorch language modeling project based on a character-level bigram model.

The main script trains a model on text data and then generates new characters autoregressively.

## Files

- `bigram.py`: Training and text generation script.
- `PrePhase.ipynb`: Notebook version -> With added MLP and Attention Mechanism in Prephase

## Requirements

- Python 3.9+
- PyTorch

Install dependencies:

```bash
pip install torch
```

## Dataset

The script expects a file named `input.txt` in this folder.

You can use Tiny Shakespeare (as referenced in the script comment):

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt
```

If `curl` is not available on your system, download the file in a browser and save it as `input.txt` in this directory.

## Run

From this folder:

```bash
python bigram.py
```

The script will:

1. Read `input.txt`.
2. Build a character vocabulary.
3. Train a simple bigram model.
4. Print training/validation loss periodically.
5. Generate sample text at the end.

## Key Hyperparameters

Defined near the top of `bigram.py`:

- `batch_size = 32`
- `block_size = 8`
- `max_iters = 3000`
- `eval_interval = 300`
- `learning_rate = 1e-2`
- `eval_iters = 200`

Device selection is automatic:

- Uses CUDA when available.
- Falls back to CPU otherwise.

## Notes

- This is a baseline educational model and not a full Transformer.
- Text quality is limited because each prediction only depends on the current token embedding.
- For better quality, increase model capacity and context length (for example, a Transformer-based architecture).

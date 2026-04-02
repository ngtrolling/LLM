import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

# ── 1. Load dataset ──────────────────────────────────────────────────────────
raw = load_dataset("roneneldan/TinyStories")
texts = raw["train"]["text"][:50_000]

# ── 2. Build character-level vocabulary ──────────────────────────────────────
corpus = "\n".join(texts)
chars  = sorted(set(corpus))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

print(f"Vocab size: {vocab_size}")

# ── 3. Encode corpus ─────────────────────────────────────────────────────────
data = torch.tensor(encode(corpus), dtype=torch.long)
n    = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ── 4. Models ─────────────────────────────────────────────────────────────────

class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding(idx)          # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs     = F.softmax(logits[:, -1, :], dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, idx_next], dim=1)
        return idx


class MLPLM(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, emb_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.block_size = block_size
        self.embedding  = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(block_size * emb_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, idx, targets=None):
        # Always take the last block_size tokens — fixes the shape mismatch during generate
        idx    = idx[:, -self.block_size:]              # (B, block_size)
        B      = idx.size(0)
        x      = self.embedding(idx)                    # (B, block_size, emb_dim)
        x      = x.view(B, -1)                         # (B, block_size * emb_dim)
        logits = self.net(x).unsqueeze(1)               # (B, 1, vocab_size)

        loss = None
        if targets is not None:
            target_last = targets[:, -1]                # predict the next token after context
            loss = F.cross_entropy(logits.squeeze(1), target_last)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop before passing in — forward also crops but this keeps things explicit
            idx_crop = idx[:, -self.block_size:]
            logits, _ = self(idx_crop)
            probs     = F.softmax(logits[:, -1, :], dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, idx_next], dim=1)
        return idx

# ── 5. Training helpers ───────────────────────────────────────────────────────
BATCH_SIZE = 64
BLOCK_SIZE = 8
EVAL_ITERS = 200
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split: str):
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([src[i : i + BLOCK_SIZE]         for i in ix])
    y   = torch.stack([src[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ── 6. Config — swap model here ───────────────────────────────────────────────
# model = BigramLM(vocab_size).to(DEVICE)
model = MLPLM(vocab_size, block_size=BLOCK_SIZE).to(DEVICE)

LR        = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

NUM_EPOCHS      = 1
steps_per_epoch = len(train_data) // (BATCH_SIZE * BLOCK_SIZE)
MAX_ITERS       = NUM_EPOCHS * steps_per_epoch
EVAL_EVERY      = 1_000

print(f"Training for {MAX_ITERS} steps (~{NUM_EPOCHS} epoch(s))")

# ── 7. Train ──────────────────────────────────────────────────────────────────
for step in range(MAX_ITERS):
    if step % EVAL_EVERY == 0:
        losses = estimate_loss(model)
        print(f"step {step:>6} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

import json
import os

SAVE_DIR = "./runs/progression_bigram"
os.makedirs(SAVE_DIR, exist_ok=True)

torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))

config = {
    "model_type": model.__class__.__name__,
    "vocab_size": vocab_size,
    "block_size": BLOCK_SIZE,
    "batch_size": BATCH_SIZE,
    "device": DEVICE,
}

with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(SAVE_DIR, "stoi.json"), "w") as f:
    json.dump(stoi, f, indent=2)

print(f"Saved model to {SAVE_DIR}")


# ── 8. Generate ───────────────────────────────────────────────────────────────
def generate_from_prompt(model, prompt: str, max_new_tokens: int = 500) -> str:
    model.eval()
    
    # Encode prompt — handle unknown chars gracefully
    encoded = [stoi[c] for c in prompt if c in stoi]
    if not encoded:
        print("Warning: no recognisable characters in prompt, starting from scratch.")
        encoded = [0]
    
    context = torch.tensor(encoded, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, T)
    generated = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return decode(generated)


prompt = "Once upon a time, there was a little girl"
print("\n── Generated text ──")
print(generate_from_prompt(model, prompt, max_new_tokens=500))
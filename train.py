"""
Training loop: AdamW, warmup + cosine LR decay, gradient clipping, loss tracking.
"""
import math
import torch
from torch.utils.data import DataLoader

from model import GPT
from dataset import load_and_split


def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float, min_lr_ratio: float = 0.1):
    """Warmup linear then cosine decay to min_lr_ratio * base_lr."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    decay = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (max_steps - warmup_steps)))
    return min_lr_ratio * base_lr + (1 - min_lr_ratio) * base_lr * decay


def train(
    block_size=128,
    batch_size=64,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_steps=5000,
    warmup_steps=200,
    lr=3e-4,
    grad_clip=1.0,
    eval_interval=200,
    save_interval=500,
    device="cuda",
):
    train_ds, val_ds, stoi, itos = load_and_split(block_size=block_size)
    vocab_size = len(stoi)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    train_iter = iter(train_loader)
    for step in range(max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        lr_step = get_lr(step, warmup_steps, max_steps, lr)
        for g in optimizer.param_groups:
            g["lr"] = lr_step

        model.train()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_losses.append(loss.item())

        if step == 0:
            print(f"step 0  initial_train_loss={loss.item():.4f}")

        if (step + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss_sum, val_n = 0.0, 0
                val_loader = DataLoader(val_ds, batch_size=batch_size)
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    _, l = model(x, y)
                    val_loss_sum += l.item() * x.size(0)
                    val_n += x.size(0)
                val_loss = val_loss_sum / val_n
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            print(f"step {step+1}/{max_steps}  train_loss={train_losses[-1]:.4f}  val_loss={val_loss:.4f}  lr={lr_step:.2e}")

        if save_interval and (step + 1) % save_interval == 0:
            ckpt = {"step": step + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_val_loss": best_val_loss}
            torch.save(ckpt, "gpt_shakespeare_ckpt.pt")

    return model, stoi, itos, train_losses, val_losses


def generate(model, itos, stoi, device, start="To be or not to ", max_new=200, temperature=0.8):
    """Autoregressive sampling: start string → encode → repeatedly predict next token and append."""
    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in start]], device=device)
    for _ in range(max_new):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_idx], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())

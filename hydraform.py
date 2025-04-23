import math
import random
import logging
import time
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _device_of(module: nn.Module) -> torch.device:
    return next(module.parameters()).device

def _clone_linear(old: nn.Linear, out_f: int) -> nn.Linear:
    new = nn.Linear(old.in_features, out_f, bias=(old.bias is not None))
    new = new.to(old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        rows = min(out_f, old.out_features)
        new.weight[:rows, :] = old.weight[:rows, :]
        if old.bias is not None:
            new.bias[:rows] = old.bias[:rows]
    return new

def _clone_linear_in(old: nn.Linear, in_f: int) -> nn.Linear:
    new = nn.Linear(in_f, old.out_features, bias=(old.bias is not None))
    new = new.to(old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        cols = min(in_f, old.in_features)
        new.weight[:, :cols] = old.weight[:, :cols]
        if old.bias is not None:
            new.bias.copy_(old.bias)
    return new

def load_data(batch_size=32, max_len=128):
    logging.info("Downloading AG News dataset…")
    ds = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    def tok(b): 
        return tokenizer(
            b["text"], padding="max_length", truncation=True, max_length=max_len
        )
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    tr = DataLoader(ds["train"], batch_size=batch_size, shuffle=True)
    te = DataLoader(ds["test"],  batch_size=batch_size)
    ncls = len(ds["train"].features["label"].names)
    logging.info(f"Train samples: {len(ds['train'])}, Test samples: {len(ds['test'])}")
    return tr, te, tokenizer.vocab_size, ncls

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="Training"):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["label"].to(device)
        logits = model(ids, mask)
        loss   = loss_fn(logits, lbl)
        loss.backward()
        if hasattr(model, "mha"):
            model.mha.track_gradients()
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item()
    return total / len(loader)

def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["label"].to(device)
            logits = model(ids, mask)
            loss   = loss_fn(logits, lbl)
            total += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == lbl).sum().item()
            count   += lbl.size(0)
    return total / len(loader), correct / count

def log_epoch_comparison(epoch, bl_metrics, ev_metrics):
    bl_tr, bl_vl, bl_acc = bl_metrics
    ev_tr, ev_vl, ev_acc = ev_metrics
    logging.info(
        f"[Epoch {epoch}] "
        f"Baseline → Train L: {bl_tr:.4f}, Val L: {bl_vl:.4f}, Acc: {bl_acc:.4f} | "
        f"Evolvable → Train L: {ev_tr:.4f}, Val L: {ev_vl:.4f}, Acc: {ev_acc:.4f}"
    )

# ---------------------------------------------------------------------
# Config & Evolvable MHA
# ---------------------------------------------------------------------

class Config:
    def __init__(
        self,
        mutation_rate:    float = 0.05,
        stochastic_depth: bool  = True,
        dropout:          float = 0.1,
        min_heads:        int   = 3,
        prune_patience:   int   = 4,
        dynamic_mutation: bool  = True,
        mutation_increase:float = 1.2,
        param_budget:     Optional[int] = 500_000,
        cooldown_epochs:  int   = 1
    ):
        self.mutation_rate     = mutation_rate
        self.stochastic_depth  = stochastic_depth
        self.dropout           = dropout
        self.min_heads         = min_heads
        self.prune_patience    = prune_patience
        self.dynamic_mutation  = dynamic_mutation
        self.mutation_increase = mutation_increase
        self.param_budget      = param_budget
        self.cooldown_epochs   = cooldown_epochs

class EvolvableAttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, mutation_rate: float, hid: int):
        super().__init__()
        self.id   = hid
        self.q    = nn.Linear(embed_dim, head_dim)
        self.k    = nn.Linear(embed_dim, head_dim)
        self.v    = nn.Linear(embed_dim, head_dim)
        self.mutation_rate = mutation_rate
        self.grad_hist: List[float] = []
        self.act_hist:  List[float] = []
        self.usage = 0
        self.drop_prob = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and random.random() < self.drop_prob:
            return torch.zeros(x.size(0), x.size(1), self.q.out_features, device=x.device, dtype=x.dtype)
        q, k, v = self.q(x), self.k(x), self.v(x)
        scale   = math.sqrt(q.size(-1))
        attn    = F.softmax(q @ k.transpose(-2, -1) / scale, dim=-1)
        out     = attn @ v
        self.usage += 1
        self.act_hist.append(out.norm().item())
        return out

    def track_gradients(self):
        total_sq = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_sq += p.grad.norm().item()**2
        self.grad_hist.append(math.sqrt(total_sq))
        if len(self.grad_hist) > 100:
            self.grad_hist.pop(0)

    def importance(self) -> float:
        if not self.grad_hist or not self.act_hist:
            return 1.0
        gs = self.grad_hist[-50:]
        as_ = self.act_hist[-50:]
        grad_score = sum(gs) / len(gs)
        act_score  = sum(as_) / len(as_)
        usage_factor = 1 + min(1.0, self.usage / 100) * 0.5
        return (grad_score + act_score) / 2 * usage_factor

    def mutate(self) -> bool:
        if len(self.grad_hist) < 20 or random.random() > self.mutation_rate:
            return False
        old_dim = self.q.out_features
        delta   = random.choice([-16, -8, 8, 16])
        new_dim = max(16, old_dim + delta)
        if new_dim == old_dim:
            return False
        self.q = _clone_linear(self.q, new_dim)
        self.k = _clone_linear(self.k, new_dim)
        self.v = _clone_linear(self.v, new_dim)
        logging.info(f"    [Head {self.id}] mutated {old_dim}→{new_dim}")
        return True

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

class EvolvableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, config: Config):
        super().__init__()
        self.embed_dim    = embed_dim
        self.config       = config
        self.next_hid     = 0
        self.cooldown     = 0
        self.val_history: List[float] = []
        self.heads        = nn.ModuleList()
        for _ in range(num_heads):
            h = EvolvableAttentionHead(embed_dim, head_dim, config.mutation_rate, self.next_hid)
            self.next_hid += 1
            self.heads.append(h)
        self._rebuild_output()
        self.dropout      = nn.Dropout(config.dropout)
        self.head_ids_history: List[List[int]] = []

    def _rebuild_output(self):
        total_dim = sum(h.q.out_features for h in self.heads)
        if hasattr(self, "out_proj"):
            self.out_proj = _clone_linear_in(self.out_proj, total_dim)
        else:
            self.out_proj = nn.Linear(total_dim, self.embed_dim).to(_device_of(self.heads[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        for h in self.heads:
            if self.config.stochastic_depth:
                h.drop_prob = (1.0 - h.importance()) * 0.5
            parts.append(h(x))
        cat = torch.cat(parts, dim=-1)
        return self.dropout(self.out_proj(cat))

    def track_gradients(self):
        for h in self.heads:
            h.track_gradients()

    def evolve(self, performance: float) -> tuple[bool, dict]:
        if self.cooldown > 0:
            self.cooldown -= 1
            return False, {}

        changed = False
        details = {}

        # Plateau detection
        self.val_history.append(performance)
        p = self.config.prune_patience
        plateau = False
        if len(self.val_history) > p:
            best_before = max(self.val_history[:-p])
            recent_best = max(self.val_history[-p:])
            plateau = recent_best <= best_before

        # Dynamic mutation-rate bump
        if self.config.dynamic_mutation and len(self.val_history) > 1:
            if performance < self.val_history[-2]:
                old_mr = self.config.mutation_rate
                self.config.mutation_rate = min(1.0, old_mr * self.config.mutation_increase)
                for h in self.heads:
                    h.mutation_rate = self.config.mutation_rate
                logging.info(f"    [MHA] mutation_rate ↑ {old_mr:.3f}→{self.config.mutation_rate:.3f}")
                changed = True
                details["mutation_rate_bumped"] = self.config.mutation_rate
                self.cooldown = self.config.cooldown_epochs

        # Mutate existing heads
        muts = []
        for h in self.heads:
            if h.mutate():
                muts.append(h.id)
        if muts:
            details["mutated"] = muts
            changed = True
            self.cooldown = self.config.cooldown_epochs

        # Add head if underperforming
        if performance < 0.5 and random.random() < self.config.mutation_rate:
            base_dim = min(h.q.out_features for h in self.heads)
            new_h = EvolvableAttentionHead(self.embed_dim, base_dim, self.config.mutation_rate, self.next_hid)
            self.next_hid += 1
            new_h.to(_device_of(self.heads[0]))
            self.heads.append(new_h)
            logging.info(f"    [MHA] added head {new_h.id} → total {len(self.heads)}")
            details.setdefault("added", []).append(new_h.id)
            changed = True
            self.cooldown = self.config.cooldown_epochs

        # Prune on plateau
        if plateau and performance > 0.8:
            rems = []
            while len(self.heads) > self.config.min_heads:
                scores = [h.importance() for h in self.heads]
                idx    = scores.index(min(scores))
                hid    = self.heads[idx].id
                self.heads.pop(idx)
                rems.append(hid)
            if rems:
                logging.info(f"    [MHA] pruned heads {rems} → total {len(self.heads)}")
                details["removed"] = rems
                changed = True
                self.cooldown = self.config.cooldown_epochs

        # Enforce param budget
        if self.config.param_budget is not None:
            pruned = []
            totp = sum(h.param_count() for h in self.heads) \
                   + sum(p.numel() for p in self.out_proj.parameters())
            while totp > self.config.param_budget and len(self.heads) > self.config.min_heads:
                scores = [h.importance() for h in self.heads]
                idx    = scores.index(min(scores))
                hid    = self.heads[idx].id
                self.heads.pop(idx)
                pruned.append(hid)
                totp = sum(h.param_count() for h in self.heads) \
                       + sum(p.numel() for p in self.out_proj.parameters())
            if pruned:
                logging.info(f"    [MHA] budget pruned heads {pruned}")
                details.setdefault("pruned", []).extend(pruned)
                changed = True
                self.cooldown = self.config.cooldown_epochs

        if changed:
            self._rebuild_output()

        self.head_ids_history.append([h.id for h in self.heads])
        return changed, details

class NewsClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        num_heads:  int,
        head_dim:   int,
        num_classes:int,
        config:     Config
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mha       = EvolvableMultiHeadAttention(embed_dim, num_heads, head_dim, config)
        self.classifier= nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.mha(x)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1)
            x = (x * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        return self.classifier(x)

# ---------------------------------------------------------------------
# Baseline Transformer
# ---------------------------------------------------------------------

class BaselineTransformer(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int,
        num_heads:   int,
        ff_dim:      int,
        num_layers:  int,
        num_classes: int,
        dropout:     float = 0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.encoder    = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        if attention_mask is not None:
            key_pad = ~attention_mask.bool()
        else:
            key_pad = None
        x = self.encoder(x, src_key_padding_mask=key_pad)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1)
            x = (x * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        return self.classifier(x)

# ---------------------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    epochs      = 3
    batch_size  = 64
    lr          = 2e-4
    embed_dim   = 128
    num_heads   = 4
    head_dim    = 32
    ff_dim      = embed_dim * 4
    num_layers  = 2
    param_budget= 500_000
    dropout     = 0.1

    # Load data
    train_dl, test_dl, vocab_size, ncls = load_data(batch_size)

    # Baseline run
    logging.info("=== Baseline Transformer ===")
    baseline = BaselineTransformer(vocab_size, embed_dim, num_heads,
                                   ff_dim, num_layers, ncls, dropout).to(device)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    bl_tr_losses, bl_val_losses, bl_val_accs = [], [], []
    for ep in range(1, epochs+1):
        logging.info(f"[Baseline] Epoch {ep}/{epochs}")
        trl = train_one_epoch(baseline, train_dl, opt_b, loss_fn, device)
        vll, vac = eval_one_epoch(baseline, test_dl, loss_fn, device)
        logging.info(f"[Baseline] Train L: {trl:.4f}, Val L: {vll:.4f}, Acc: {vac:.4f}")
        bl_tr_losses.append(trl)
        bl_val_losses.append(vll)
        bl_val_accs.append(vac)

    # Free baseline
    del baseline, opt_b
    torch.cuda.empty_cache()

    # Evolvable run
    logging.info("=== Evolvable Transformer ===")
    cfg = Config(
        mutation_rate=0.1,
        stochastic_depth=True,
        dropout=dropout,
        min_heads=3,
        prune_patience=4,
        dynamic_mutation=True,
        mutation_increase=1.3,
        param_budget=param_budget,
        cooldown_epochs=1
    )
    evolvable = NewsClassifier(vocab_size, embed_dim, num_heads,
                               head_dim, ncls, cfg).to(device)
    opt_e = torch.optim.Adam(evolvable.parameters(), lr=lr)

    ev_tr_losses, ev_val_losses, ev_val_accs = [], [], []
    ev_details_history = []
    evolvable.mha.head_ids_history.append([h.id for h in evolvable.mha.heads])

    for ep in range(1, epochs+1):
        logging.info(f"[Evolvable] Epoch {ep}/{epochs}")
        trl = train_one_epoch(evolvable, train_dl, opt_e, loss_fn, device)
        vll, vac = eval_one_epoch(evolvable, test_dl, loss_fn, device)
        logging.info(f"[Evolvable] Train L: {trl:.4f}, Val L: {vll:.4f}, Acc: {vac:.4f}")

        bl_metrics = (bl_tr_losses[ep-1], bl_val_losses[ep-1], bl_val_accs[ep-1])
        ev_metrics = (trl, vll, vac)
        log_epoch_comparison(ep, bl_metrics, ev_metrics)

        changed, details = evolvable.mha.evolve(vac)
        ev_details_history.append(details)
        if changed:
            logging.info(f"[Evolvable] → Evolved: {details}")
        else:
            logging.info("[Evolvable] → No evolution")

    # Plot & report
    ts = time.strftime("%Y%m%d_%H%M%S")
    epochs_range = list(range(1, epochs+1))

    # Loss curves
    plt.figure(figsize=(10,4))
    plt.plot(epochs_range, bl_tr_losses,   label="Baseline Train")
    plt.plot(epochs_range, bl_val_losses,  label="Baseline Val")
    plt.plot(epochs_range, ev_tr_losses,   label="Evo Train")
    plt.plot(epochs_range, ev_val_losses,  label="Evo Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss: Baseline vs Evolvable")
    plt.legend(); plt.grid(True); plt.tight_layout()
    loss_fn_name = f"comparison_loss_{ts}.png"
    plt.savefig(loss_fn_name)

    # Accuracy curves
    plt.figure(figsize=(10,4))
    plt.plot(epochs_range, bl_val_accs, label="Baseline Acc")
    plt.plot(epochs_range, ev_val_accs, label="Evolvable Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Val Accuracy: Baseline vs Evolvable")
    plt.legend(); plt.grid(True); plt.tight_layout()
    acc_fn_name = f"comparison_acc_{ts}.png"
    plt.savefig(acc_fn_name)

    # Head lineage tree
    unique_ids = sorted({hid for hist in evolvable.mha.head_ids_history for hid in hist})
    id_to_y = {hid:i for i,hid in enumerate(unique_ids)}
    plt.figure(figsize=(10, 1 + len(unique_ids)*0.5))
    for hid in unique_ids:
        xs = [e+1 for e,hist in enumerate(evolvable.mha.head_ids_history) if hid in hist]
        ys = [id_to_y[hid]] * len(xs)
        plt.plot(xs, ys, marker="o")
    plt.yticks(list(id_to_y.values()), [f"H{hid}" for hid in unique_ids])
    plt.xlabel("Epoch"); plt.title("Evolvable Head Lineage")
    plt.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
    lineage_fn = f"comparison_head_lineage_{ts}.png"
    plt.savefig(lineage_fn)

    # Write TXT report
    report_name = f"report_{ts}.txt"
    with open(report_name, "w") as f:
        f.write("Baseline vs Evolvable Transformer Report\n")
        f.write("="*50 + "\n\n")
        for ep in range(1, epochs+1):
            f.write(f"Epoch {ep}:\n")
            f.write(f"  Baseline   → Train L: {bl_tr_losses[ep-1]:.4f}, "
                    f"Val L: {bl_val_losses[ep-1]:.4f}, Acc: {bl_val_accs[ep-1]:.4f}\n")
            f.write(f"  Evolvable  → Train L: {ev_tr_losses[ep-1]:.4f}, "
                    f"Val L: {ev_val_losses[ep-1]:.4f}, Acc: {ev_val_accs[ep-1]:.4f}\n")
            if ev_details_history[ep-1]:
                f.write(f"    Evolution events: {ev_details_history[ep-1]}\n")
            f.write("\n")
        f.write("Diagnostics Visuals:\n")
        f.write(f"  Loss curves: {loss_fn_name}\n")
        f.write(f"  Accuracy:    {acc_fn_name}\n")
        f.write(f"  Head lineage: {lineage_fn}\n")
    logging.info(f"Plots and report generated: {loss_fn_name}, {acc_fn_name}, {lineage_fn}, {report_name}")

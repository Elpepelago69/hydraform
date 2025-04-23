"""
MIT License
Author: github.com/tegridydev/hydraform

Hydraform [Self-Evolving Attention]

Think of a classical Hydra, where individual heads grow, shrink or regenerate
based on performance.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage"
)

import math
import random
import time
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.progress import track
from rich.panel import Panel
from rich import box
import matplotlib.pyplot as plt

console = Console()


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


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_data(batch_size=32, max_len=128):
    console.log("📥 Downloading AG News dataset…")
    ds = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    def tok(batch):
        return tokenizer(batch["text"],
                         padding="max_length",
                         truncation=True,
                         max_length=max_len)
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    train_dl = DataLoader(ds["train"], batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(ds["test"],  batch_size=batch_size)
    num_labels = len(ds["train"].features["label"].names)
    console.log(f"[green]Train:[/green] {len(ds['train'])}  [green]Test:[/green] {len(ds['test'])}")
    return train_dl, test_dl, tokenizer.vocab_size, num_labels


# ---------------------------------------------------------------------
# Train/Eval loops
# ---------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in track(loader, description="Training"):
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
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for batch in track(loader, description="Evaluating"):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["label"].to(device)
        with torch.no_grad():
            logits = model(ids, mask)
            loss   = loss_fn(logits, lbl)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == lbl).sum().item()
        count   += lbl.size(0)
    return total_loss / len(loader), correct / count


def log_epoch_comparison(epoch, bl_metrics, ev_metrics):
    bl_tr, bl_vl, bl_acc = bl_metrics
    ev_tr, ev_vl, ev_acc = ev_metrics
    console.log(
        f"[bold]Epoch {epoch}[/bold]  "
        f"[cyan]Baseline[/cyan]→ L:{bl_vl:.4f} Acc:{bl_acc:.4f} | "
        f"[magenta]Evolvable[/magenta]→ L:{ev_vl:.4f} Acc:{ev_acc:.4f}"
    )


# ---------------------------------------------------------------------
# Model Definitions
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
    def __init__(self, embed_dim, head_dim, mutation_rate, hid):
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

    def forward(self, x):
        if self.training and random.random() < self.drop_prob:
            return torch.zeros(x.size(0), x.size(1), self.q.out_features,
                               device=x.device, dtype=x.dtype)
        q,k,v = self.q(x), self.k(x), self.v(x)
        scale = math.sqrt(q.size(-1))
        attn  = F.softmax(q @ k.transpose(-2,-1)/scale, dim=-1)
        out   = attn @ v
        self.usage += 1
        self.act_hist.append(out.norm().item())
        return out

    def track_gradients(self):
        total_sq = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_sq += p.grad.norm().item()**2
        self.grad_hist.append(math.sqrt(total_sq))
        if len(self.grad_hist)>100:
            self.grad_hist.pop(0)

    def importance(self):
        if not self.grad_hist or not self.act_hist:
            return 1.0
        gs = self.grad_hist[-50:];   as_ = self.act_hist[-50:]
        grad_score = sum(gs)/len(gs)
        act_score  = sum(as_)/len(as_)
        usage_factor = 1 + min(1.0,self.usage/100)*0.5
        return (grad_score+act_score)/2 * usage_factor

    def mutate(self):
        if len(self.grad_hist)<20 or random.random()>self.mutation_rate:
            return False
        old_dim = self.q.out_features
        delta   = random.choice([-16,-8,8,16])
        new_dim = max(16, old_dim+delta)
        if new_dim==old_dim:
            return False
        self.q = _clone_linear(self.q, new_dim)
        self.k = _clone_linear(self.k, new_dim)
        self.v = _clone_linear(self.v, new_dim)
        console.log(f"    [yellow]Head {self.id}[/yellow] mutated {old_dim}→{new_dim}")
        return True

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

class EvolvableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim, config: Config):
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

    def forward(self, x):
        parts = []
        for h in self.heads:
            if self.config.stochastic_depth:
                h.drop_prob = (1.0 - h.importance())*0.5
            parts.append(h(x))
        cat = torch.cat(parts, dim=-1)
        return self.dropout(self.out_proj(cat))

    def track_gradients(self):
        for h in self.heads:
            h.track_gradients()

    def evolve(self, performance: float):
        if self.cooldown>0:
            self.cooldown -=1
            return False, {}

        changed = False
        details = {}

        # plateau detection
        self.val_history.append(performance)
        p = self.config.prune_patience
        plateau = False
        if len(self.val_history) > p:
            best_before = max(self.val_history[:-p])
            recent_best = max(self.val_history[-p:])
            plateau = recent_best <= best_before

        # dynamic mutation bump
        if self.config.dynamic_mutation and len(self.val_history)>1:
            if performance < self.val_history[-2]:
                old_mr = self.config.mutation_rate
                self.config.mutation_rate = min(1.0, old_mr*self.config.mutation_increase)
                for h in self.heads:
                    h.mutation_rate = self.config.mutation_rate
                console.log(f"    [cyan]mutation_rate[/cyan] ↑ {old_mr:.3f}→{self.config.mutation_rate:.3f}")
                changed=True
                details["mutation_rate_bumped"]=self.config.mutation_rate
                self.cooldown=self.config.cooldown_epochs

        # mutate heads
        muts=[]
        for h in self.heads:
            if h.mutate():
                muts.append(h.id)
        if muts:
            details["mutated"]=muts
            changed=True
            self.cooldown=self.config.cooldown_epochs

        # add head
        if performance<0.5 and random.random()<self.config.mutation_rate:
            base=min(h.q.out_features for h in self.heads)
            new_h=EvolvableAttentionHead(self.embed_dim, base, self.config.mutation_rate, self.next_hid)
            self.next_hid+=1
            new_h.to(_device_of(self.heads[0]))
            self.heads.append(new_h)
            console.log(f"    [green]added head[/green] {new_h.id} (total {len(self.heads)})")
            details.setdefault("added",[]).append(new_h.id)
            changed=True
            self.cooldown=self.config.cooldown_epochs

        # prune on plateau
        if plateau and performance>0.8:
            rems=[]
            while len(self.heads)>self.config.min_heads:
                scores=[h.importance() for h in self.heads]
                idx=scores.index(min(scores))
                hid=self.heads[idx].id
                self.heads.pop(idx)
                rems.append(hid)
            if rems:
                console.log(f"    [red]pruned heads[/red] {rems}")
                details["removed"]=rems
                changed=True
                self.cooldown=self.config.cooldown_epochs

        # param budget
        if self.config.param_budget is not None:
            pruned=[]
            totp=sum(h.param_count() for h in self.heads)+sum(p.numel() for p in self.out_proj.parameters())
            while totp>self.config.param_budget and len(self.heads)>self.config.min_heads:
                scores=[h.importance() for h in self.heads]
                idx=scores.index(min(scores))
                hid=self.heads[idx].id
                self.heads.pop(idx)
                pruned.append(hid)
                totp=sum(h.param_count() for h in self.heads)+sum(p.numel() for p in self.out_proj.parameters())
            if pruned:
                console.log(f"    [red]budget prune[/red] {pruned}")
                details.setdefault("pruned",[]).extend(pruned)
                changed=True
                self.cooldown=self.config.cooldown_epochs

        if changed:
            self._rebuild_output()

        self.head_ids_history.append([h.id for h in self.heads])
        return changed, details

class NewsClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, head_dim, num_classes, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mha       = EvolvableMultiHeadAttention(embed_dim, num_heads, head_dim, config)
        self.classifier= nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.mha(x)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1)
            x = (x*m).sum(1)/m.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        return self.classifier(x)

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, num_classes, dropout=0.1):
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
        key_pad = (~attention_mask.bool()) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_pad)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1)
            x = (x*m).sum(1)/m.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        return self.classifier(x)


# ---------------------------------------------------------------------
# Run routines
# ---------------------------------------------------------------------

def run_baseline(train_dl, test_dl, vocab_size, ncls, device,
                 epochs, lr, embed_dim, num_heads, ff_dim, num_layers, dropout):
    model = BaselineTransformer(vocab_size, embed_dim, num_heads,
                                ff_dim, num_layers, ncls, dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    bl_tr, bl_vl, bl_acc = [], [], []
    console.log(Panel("➡️  Running Baseline Transformer", box=box.ROUNDED))
    for ep in range(1, epochs+1):
        console.log(f"[cyan]Baseline Epoch {ep}/{epochs}[/cyan]")
        trl = train_one_epoch(model, train_dl, opt, loss_fn, device)
        vll, vac = eval_one_epoch(model, test_dl, loss_fn, device)
        console.log(f"  [cyan]Train L:[/cyan]{trl:.4f}  [cyan]Val L:[/cyan]{vll:.4f}  [cyan]Acc:[/cyan]{vac:.4f}")
        bl_tr.append(trl); bl_vl.append(vll); bl_acc.append(vac)
    return bl_tr, bl_vl, bl_acc

def run_evolvable(train_dl, test_dl, vocab_size, ncls, device,
                  epochs, lr, embed_dim, num_heads, head_dim, cfg,
                  baseline_metrics=None):
    model = NewsClassifier(vocab_size, embed_dim, num_heads, head_dim, ncls, cfg).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    ev_tr, ev_vl, ev_acc, ev_details = [], [], [], []
    model.mha.head_ids_history.append([h.id for h in model.mha.heads])
    console.log(Panel("➡️  Running Evolvable Transformer", box=box.ROUNDED))
    for ep in range(1, epochs+1):
        console.log(f"[magenta]Evolvable Epoch {ep}/{epochs}[/magenta]")
        trl = train_one_epoch(model, train_dl, opt, loss_fn, device)
        vll, vac = eval_one_epoch(model, test_dl, loss_fn, device)
        console.log(f"  [magenta]Train L:[/magenta]{trl:.4f}  [magenta]Val L:[/magenta]{vll:.4f}  [magenta]Acc:[/magenta]{vac:.4f}")
        if baseline_metrics:
            bl_tr, bl_vl, bl_acc = baseline_metrics
            log_epoch_comparison(ep, (bl_tr[ep-1], bl_vl[ep-1], bl_acc[ep-1]), (trl, vll, vac))
        ev_tr.append(trl); ev_vl.append(vll); ev_acc.append(vac)
        changed, details = model.mha.evolve(vac)
        ev_details.append(details)
        if changed:
            console.log(f"    [green]Evolved:[/green] {details}")
        else:
            console.log("    [yellow]No evolution[/yellow]")
    return ev_tr, ev_vl, ev_acc, ev_details, model.mha.head_ids_history

def plot_and_report(bl, ev, details, head_hist, ts):
    bl_tr, bl_vl, bl_acc = bl
    ev_tr, ev_vl, ev_acc = ev

    x_bl = list(range(1, len(bl_tr)+1))
    x_ev = list(range(1, len(ev_tr)+1))

    # 1) Loss
    plt.figure(figsize=(10,4))
    if bl_tr:
        plt.plot(x_bl, bl_tr,  label="Baseline Train")
        plt.plot(x_bl, bl_vl,  label="Baseline Val")
    if ev_tr:
        plt.plot(x_ev, ev_tr,  label="Evo Train")
        plt.plot(x_ev, ev_vl,  label="Evo Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss: Baseline vs Evolvable")
    plt.legend(); plt.grid(True); plt.tight_layout()
    loss_fn_name = f"comparison_loss_{ts}.png"
    plt.savefig(loss_fn_name)

    # 2) Accuracy
    plt.figure(figsize=(10,4))
    if bl_acc:
        plt.plot(x_bl, bl_acc, label="Baseline Acc")
    if ev_acc:
        plt.plot(x_ev, ev_acc, label="Evolvable Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Val Accuracy: Baseline vs Evolvable")
    plt.legend(); plt.grid(True); plt.tight_layout()
    acc_fn_name = f"comparison_acc_{ts}.png"
    plt.savefig(acc_fn_name)

    # 3) Head lineage
    if head_hist:
        unique_ids = sorted({hid for hist in head_hist for hid in hist})
        id_to_y    = {hid:i for i,hid in enumerate(unique_ids)}
        plt.figure(figsize=(10, 1+len(unique_ids)*0.5))
        for hid in unique_ids:
            xs = [e+1 for e,hist in enumerate(head_hist) if hid in hist]
            ys = [id_to_y[hid]] * len(xs)
            plt.plot(xs, ys, marker="o")
        plt.yticks(list(id_to_y.values()), [f"H{hid}" for hid in unique_ids])
        plt.xlabel("Epoch"); plt.title("Head Lineage")
        plt.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
        lineage_fn = f"comparison_head_lineage_{ts}.png"
        plt.savefig(lineage_fn)
    else:
        lineage_fn = None

    # 4) TXT report
    report_name = f"report_{ts}.txt"
    with open(report_name, "w", encoding="utf-8") as f:
        f.write("Hydraform Benchmark Report\n")
        f.write("="*50 + "\n\n")
        epochs = max(len(bl_tr), len(ev_tr))
        for i in range(epochs):
            f.write(f"Epoch {i+1}:\n")
            if i < len(bl_tr):
                f.write(f"  Baseline   -> Train L: {bl_tr[i]:.4f}, Val L: {bl_vl[i]:.4f}, Acc: {bl_acc[i]:.4f}\n")
            else:
                f.write("  Baseline   -> (no data)\n")
            if i < len(ev_tr):
                f.write(f"  Evolvable  -> Train L: {ev_tr[i]:.4f}, Val L: {ev_vl[i]:.4f}, Acc: {ev_acc[i]:.4f}\n")
                if i < len(details) and details[i]:
                    f.write(f"    Events: {details[i]}\n")
            else:
                f.write("  Evolvable  -> (no data)\n")
            f.write("\n")
        f.write("Generated Plots:\n")
        f.write(f"  Loss curve:    {loss_fn_name}\n")
        f.write(f"  Accuracy plot: {acc_fn_name}\n")
        if lineage_fn:
            f.write(f"  Head lineage:  {lineage_fn}\n")
    console.log(f"✅ Saved: {loss_fn_name}, {acc_fn_name}" +
                (f", {lineage_fn}" if lineage_fn else "") +
                f", {report_name}")


# ---------------------------------------------------------------------
# Main Menu
# ---------------------------------------------------------------------

def main_menu():
    console.clear()
    console.print(Panel("🔱 [bold cyan]Hydraform[/bold cyan]\n"
                        "Self-Evolving Transformer Benchmark",
                        box=box.ROUNDED))

    # Hyperparameter prompts
    epochs      = IntPrompt.ask("Number of epochs", default=5)
    batch_size  = IntPrompt.ask("Batch size", default=64)
    lr          = float(Prompt.ask("Learning rate", default="2e-4"))
    embed_dim   = IntPrompt.ask("Embedding dim", default=128)
    num_heads   = IntPrompt.ask("Initial # heads", default=4)
    head_dim    = IntPrompt.ask("Head dim", default=32)
    num_layers  = IntPrompt.ask("Baseline layers", default=2)
    param_budget= IntPrompt.ask("Param budget (0 to disable)", default=500000)
    if param_budget <= 0:
        param_budget = None

    cfg = Config(
        mutation_rate     = float(Prompt.ask("Mutation rate", default="0.1")),
        stochastic_depth  = Confirm.ask("Use stochastic depth?", default=True),
        dropout           = float(Prompt.ask("Dropout", default="0.1")),
        min_heads         = IntPrompt.ask("Min heads", default=3),
        prune_patience    = IntPrompt.ask("Prune patience", default=4),
        dynamic_mutation  = Confirm.ask("Dynamic mutation bump?", default=True),
        mutation_increase = float(Prompt.ask("Mutation increase factor", default="1.3")),
        param_budget      = param_budget,
        cooldown_epochs   = IntPrompt.ask("Cooldown epochs", default=1)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, test_dl, vocab_size, ncls = load_data(batch_size)

    while True:
        console.print("\n[bold]Menu:[/bold]")
        console.print("1) Run Baseline only")
        console.print("2) Run Evolvable only")
        console.print("3) Run Full Benchmark")
        console.print("4) Exit")
        choice = Prompt.ask("Choice", choices=["1","2","3","4"])

        ts = time.strftime("%Y%m%d_%H%M%S")
        if choice == "1":
            bl = run_baseline(train_dl, test_dl, vocab_size, ncls, device,
                              epochs, lr, embed_dim, num_heads,
                              embed_dim*4, num_layers, cfg.dropout)
            plot_and_report(bl, ([],[],[]), [], [], ts)

        elif choice == "2":
            ev = run_evolvable(train_dl, test_dl, vocab_size, ncls, device,
                               epochs, lr, embed_dim, num_heads, head_dim, cfg)
            plot_and_report(([],[],[]), ev[:3], ev[3], ev[4], ts)

        elif choice == "3":
            bl = run_baseline(train_dl, test_dl, vocab_size, ncls, device,
                              epochs, lr, embed_dim, num_heads,
                              embed_dim*4, num_layers, cfg.dropout)
            ev = run_evolvable(train_dl, test_dl, vocab_size, ncls, device,
                               epochs, lr, embed_dim, num_heads, head_dim, cfg,
                               baseline_metrics=(bl[0], bl[1], bl[2]))
            plot_and_report(bl, ev[:3], ev[3], ev[4], ts)

        else:
            console.print("👋 Goodbye!")
            break


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user, exiting...[/red]")

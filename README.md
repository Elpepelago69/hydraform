# Hydraform [Self-Evolving Attention]

Hydraform currently consists of a PoC benchmarking toolkit and research prototype for self-evolving, mutation-guided multi-headed attention.

*Think of a classical Hydra, where individual heads grow, shrink or regenerate based on performance.*

It compares two models on the AG News text classification task:

1. **Baseline**: a fixed `nn.TransformerEncoder`  
2. **Evolvable**: a Transformer whose multi-head attention block can dynamically mutate head dimensions, add/remove heads, and prune under a parameter budget  

Each run produces:  
- Timestamped **loss & accuracy comparison** plots  
- A **head-lineage tree** visualizing how each attention head persists or changes  
- A  **`.txt` report** of per-epoch metrics and evolution events

## Set Up / Install

```bash
git clone https://github.com/tegridydev/hydraform.git
cd hydraform
pip install -r requirements.txt
```

## Run
```bash
python hydraform.py
```

Outputs will appear in your working directory, e.g.:

- comparison_loss_<timestamp>.png

- comparison_acc_<timestamp>.png

- comparison_head_lineage_<timestamp>.png

- report_<timestamp>.txt

# DualCLF: Dual Contrastive Learning for AI-Generated Text Detection

Code for the paper:

> **BHAID: A Dataset of LLM-Generated Text in Indian Languages with a Dual Contrastive Learning Framework for Detection**
> Anonymous submission (under review)

DualCLF combines a self-supervised contrastive loss, a supervised contrastive loss, and focal loss on top of an XLM-RoBERTa encoder:

```
L_total = L_FL + λ (L_self + L_sup)
```

## Project structure

```
DualCLF/
├── main.py            # entry point (all modes)
├── requirements.txt
├── README.md
└── src/
    ├── data.py        # BHAID loading, 70/10/20 split, batching
    ├── model.py       # encoder + projection head + classifier, token-dropout augmentation
    ├── losses.py      # NT-Xent, supervised contrastive, focal loss
    ├── engine.py      # training, evaluation, plots
    └── utils.py
```

## Dataset

BHAID covers nine Indian languages (Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Odia, Assamese). Fields: `Text`, `Label` (`human` / `ai`), `Lang`.

`main.py` loads it directly from the Hugging Face Hub:

```python
from datasets import load_dataset
ds = load_dataset("Sifi-world/BHAID", data_files="final_lang.parquet")
```

To use a local copy instead: `python main.py --local_file final_lang.parquet`.

The split is 70% train / 10% validation / 20% test, stratified by language and label (seed 42).

## Installation

```bash
git clone <REPO_URL>
cd DualCLF
pip install -r requirements.txt
```

## Usage

```bash
# Full DualCLF (L_FL + λ(L_self + L_sup))
python main.py

# Ablations
python main.py --mode supervised          # L_sup + L_FL
python main.py --mode self_supervised     # L_self only

# Cross-lingual: train on Gujarati, test on Hindi
python main.py --train_langs Gu --test_langs Hi

# Options
python main.py --model_name xlm-roberta-base --epochs 10 --batch_size 32 --fp16
```

Run `python main.py --help` for all options.

## Default hyperparameters

These follow Table 18 of the paper.

| Parameter | Value |
|-----------|-------|
| Encoder | `xlm-roberta-base` |
| Optimizer | AdamW, lr 2e-5, weight decay 0.01 |
| Batch size | 32 |
| Epochs | 10 |
| Warmup ratio | 0.1 |
| Dropout | 0.3 |
| Temperature τ | 0.07 |
| Focal loss | γ = 2, α = [0.75, 0.25] (human, AI) |
| λ | 0.3 |
| Max length | 256 |
| Augmentation | token dropout, p = 0.10 |
| Gradient clipping | 1.0 |

The best checkpoint is chosen by validation macro-F1 (silhouette score for `self_supervised`) and evaluated once on the test set.

## Outputs

Saved to `outputs/<mode>/`:

- `model.pt`: best checkpoint
- `results.json`: test accuracy, macro/micro F1, F1 per class, AUROC, and per-language scores
- `history.json`: per-epoch losses and validation metrics
- `losses.png`, `metrics_over_epochs.png`, `separation_over_epochs.png`

## Inference

Only the encoder and classification head are used at inference. The projection head and contrastive losses are not needed.

## Limitations

Performance on BHAID does not show that a detector works across all domains, languages, or generators. Detectors can produce false positives on genuine human writing, so they should not be the only basis for decisions about individual authors.

## Citation

```bibtex
@inproceedings{bhaid2026,
  title     = {BHAID: A Dataset of LLM-Generated Text in Indian Languages with a Dual Contrastive Learning Framework for Detection},
  author    = {Anonymous},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics},
  year      = {2026}
}
```

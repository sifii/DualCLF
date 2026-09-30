#!/usr/bin/env python
"""DualCLF entry point.

Examples
--------
python main.py                                  # full DualCLF on BHAID
python main.py --mode supervised                # L_sup + L_FL ablation
python main.py --mode self_supervised           # L_self ablation
python main.py --train_langs Gu --test_langs Hi # cross-lingual
python main.py --local_file final_lang.parquet  # use a local copy
"""
import argparse

from src.engine import run


def get_args():
    p = argparse.ArgumentParser(description="DualCLF on BHAID")
    # data
    p.add_argument("--dataset", default="Sifi-world/BHAID")
    p.add_argument("--data_file", default="final_lang.parquet")
    p.add_argument("--local_file", default=None, help="Local parquet; skips the Hub")
    p.add_argument("--train_langs", default=None, help="Comma list, e.g. Gu,Hi")
    p.add_argument("--test_langs", default=None, help="Comma list; default = train_langs")
    p.add_argument("--max_len", type=int, default=256)
    # mode
    p.add_argument("--mode", choices=["dual", "supervised", "self_supervised"], default="dual")
    # model
    p.add_argument("--model_name", default="xlm-roberta-base")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    # optimisation (defaults follow Table 18 of the paper)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lambda_", type=float, default=0.3, help="Weight of contrastive terms")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--focal_alpha", type=float, nargs=2, default=[0.75, 0.25],
                   help="Class weights [human, ai]")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--aug_drop_prob", type=float, default=0.10)
    p.add_argument("--balanced_sampler", action="store_true")
    p.add_argument("--fp16", action="store_true", help="Mixed precision (CUDA only)")
    # misc
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    run(get_args())

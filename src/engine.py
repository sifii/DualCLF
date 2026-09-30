"""Training and evaluation."""
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import TextDataset, load_bhaid, make_collate, make_splits
from .losses import FocalLoss, nt_xent, supervised_contrastive
from .model import DualCLFModel, dropout_augment
from .utils import save_json, set_seed

SIL_MAX = 5000  # silhouette is O(n^2); subsample for speed


@torch.no_grad()
def evaluate(model, loader, device, langs=None, with_cls=True):
    model.eval()
    zs, logits_all, labels_all = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        z, logits = model(ids, mask)
        zs.append(z.float().cpu())
        logits_all.append(logits.float().cpu())
        labels_all.append(batch["labels"])
    z = torch.cat(zs).numpy()
    logits = torch.cat(logits_all)
    y = torch.cat(labels_all).numpy()

    res = {}
    try:
        idx = np.random.RandomState(0).permutation(len(y))[:SIL_MAX]
        res["silhouette"] = float(silhouette_score(z[idx], y[idx], metric="cosine"))
    except Exception:
        res["silhouette"] = 0.0
    if not with_cls:
        return res

    pred = logits.argmax(-1).numpy()
    res.update({
        "acc": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y, pred, average="micro", zero_division=0),
        "f1_human": f1_score(y, pred, pos_label=0, average="binary", zero_division=0),
        "f1_ai": f1_score(y, pred, pos_label=1, average="binary", zero_division=0),
    })
    try:
        res["auroc"] = float(roc_auc_score(y, F.softmax(logits, -1)[:, 1].numpy()))
    except Exception:
        res["auroc"] = float("nan")
    res["pred_dist"] = dict(Counter(pred.tolist()))

    if langs is not None:
        langs = np.array(langs)
        per = {}
        for l in sorted(set(langs)):
            m = langs == l
            per[l] = {"n": int(m.sum()),
                      "acc": accuracy_score(y[m], pred[m]),
                      "f1_macro": f1_score(y[m], pred[m], average="macro", zero_division=0)}
        res["per_language"] = per
    return res


def _plot(history, out_dir, mode):
    ep = history["epoch"]
    plt.figure(figsize=(8, 4))
    for k, lab in [("loss_focal", "Focal"), ("loss_se", "NT-Xent"), ("loss_su", "SupCon")]:
        if any(v != 0 for v in history[k]):
            plt.plot(ep, history[k], marker="o", label=lab)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training losses")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "losses.png"), dpi=200); plt.close()

    if mode != "self_supervised":
        plt.figure(figsize=(8, 4))
        for k, lab in [("val_acc", "Accuracy"), ("val_f1_macro", "Macro-F1"), ("val_f1_micro", "Micro-F1")]:
            plt.plot(ep, history[k], marker="o", label=lab)
        plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.title("Validation metrics")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "metrics_over_epochs.png"), dpi=200); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ep, history["val_sil"], marker="o", color="purple")
    plt.xlabel("Epoch"); plt.ylabel("Silhouette"); plt.title("Embedding separation")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "separation_over_epochs.png"), dpi=200); plt.close()


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    out_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device} | mode: {args.mode} | encoder: {args.model_name}")

    # ---------------- data ----------------
    df = load_bhaid(args.dataset, args.data_file, args.local_file)
    print(f"Loaded {len(df)} samples")
    print(df.groupby(["Lang", "Label"]).size().unstack(fill_value=0))
    splits = make_splits(
        df, args.seed,
        train_langs=args.train_langs.split(",") if args.train_langs else None,
        test_langs=args.test_langs.split(",") if args.test_langs else None,
    )
    for k, v in splits.items():
        print(f"  {k}: {len(v)} (human={int((v.y == 0).sum())}, ai={int((v.y == 1).sum())})")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    collate = make_collate(tok, args.max_len)
    train_ds, val_ds, test_ds = (TextDataset(splits[k]) for k in ("train", "validation", "test"))

    sampler = None
    if args.balanced_sampler:
        cnt = np.bincount(train_ds.labels)
        w = [1.0 / cnt[l] for l in train_ds.labels]
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          shuffle=sampler is None, collate_fn=collate, drop_last=True,
                          num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.eval_batch_size, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=args.eval_batch_size, collate_fn=collate)

    # ---------------- model ----------------
    model = DualCLFModel(args.model_name, args.proj_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    focal = FocalLoss(args.focal_alpha, args.focal_gamma).to(device)
    use_amp = args.fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with_cls = args.mode != "self_supervised"
    key = "f1_macro" if with_cls else "silhouette"
    keys = ["epoch", "loss_focal", "loss_se", "loss_su", "val_acc", "val_f1_macro",
            "val_f1_micro", "val_auroc", "val_sil"]
    history = {k: [] for k in keys}
    best, ckpt = -1e9, os.path.join(out_dir, "model.pt")

    # ---------------- train ----------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = {"f": 0.0, "se": 0.0, "su": 0.0}
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                ids1 = dropout_augment(ids, mask, tok.pad_token_id, args.aug_drop_prob)
                z1, logits = model(ids1, mask)
                lf = lse = lsu = torch.zeros((), device=device)
                if args.mode in ("dual", "self_supervised"):
                    ids2 = dropout_augment(ids, mask, tok.pad_token_id, args.aug_drop_prob)
                    z2, _ = model(ids2, mask)
            z1f = z1.float()
            if args.mode in ("dual", "supervised"):
                lf = focal(logits.float(), y)
            if args.mode in ("dual", "self_supervised"):
                lse = torch.clamp(nt_xent(z1f, z2.float(), args.temperature), max=10.0)
            if args.mode in ("dual", "supervised"):
                lsu = torch.clamp(supervised_contrastive(z1f, y, args.temperature), max=10.0)

            if args.mode == "dual":
                loss = lf + args.lambda_ * (lse + lsu)
            elif args.mode == "supervised":
                loss = lf + args.lambda_ * lsu
            else:
                loss = lse

            if not torch.isfinite(loss):
                print("Skipping non-finite loss batch")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot["f"] += lf.item(); tot["se"] += lse.item(); tot["su"] += lsu.item()

        n = max(len(train_dl), 1)
        val = evaluate(model, val_dl, device, with_cls=with_cls)
        history["epoch"].append(epoch)
        history["loss_focal"].append(tot["f"] / n)
        history["loss_se"].append(tot["se"] / n)
        history["loss_su"].append(tot["su"] / n)
        history["val_acc"].append(val.get("acc", 0.0))
        history["val_f1_macro"].append(val.get("f1_macro", 0.0))
        history["val_f1_micro"].append(val.get("f1_micro", 0.0))
        history["val_auroc"].append(val.get("auroc", float("nan")))
        history["val_sil"].append(val["silhouette"])
        print(f"[Epoch {epoch:02d}] focal={tot['f']/n:.4f} se={tot['se']/n:.4f} su={tot['su']/n:.4f} | "
              + " ".join(f"{k}={v:.4f}" for k, v in val.items() if isinstance(v, float)))

        if val[key] > best:
            best = val[key]
            torch.save(model.state_dict(), ckpt)
            print(f"  saved best model ({key}={best:.4f})")

    # ---------------- test ----------------
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate(model, test_dl, device, langs=test_ds.langs, with_cls=with_cls)
    print("\n[TEST]", {k: v for k, v in test.items() if k != "per_language"})
    save_json({"best_val_" + key: best, "test": test, "args": vars(args)},
              os.path.join(out_dir, "results.json"))
    save_json(history, os.path.join(out_dir, "history.json"))
    _plot(history, out_dir, args.mode)
    print(f"Done. Outputs in {out_dir}")

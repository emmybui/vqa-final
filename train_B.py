"""
Training script – Hướng B (BLIP fine-tuning)
Usage:
    python train_B.py --mode finetune   # B2
    python train_B.py --mode zero_shot  # B1 (evaluation only)
"""

import os, sys, argparse, time, json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from models.model_B import BLIPFineTuned, BLIPZeroShot


# ─────────────────────────────────────────────────────────────────────────────
# BLIP Dataset (PIL image + raw text, no vocab needed)
# ─────────────────────────────────────────────────────────────────────────────

class BLIPVQADataset(Dataset):
    def __init__(self, csv_path: str, img_base_dir: str = None, split: str = "train"):
        self.df = pd.read_csv(csv_path)
        if len(self.df.columns) == 1:
            self.df = pd.read_csv(csv_path, sep="\t")
        self.img_base = img_base_dir
        self.split = split

    def _resolve(self, raw_path):
        if self.img_base:
            fname = os.path.basename(raw_path)
            for root, _, files in os.walk(self.img_base):
                if fname in files:
                    return os.path.join(root, fname)
        return raw_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._resolve(str(row["image_path"]))
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        return {
            "image":      image,
            "question":   str(row["question"]),
            "answer":     str(row["answer"]),
            "raw_answer": str(row["answer"]),
        }


def collate_pil(batch):
    """Custom collate: PIL images cannot be stacked."""
    images    = [b["image"]    for b in batch]
    questions = [b["question"] for b in batch]
    answers   = [b["answer"]   for b in batch]
    raw_ans   = [b["raw_answer"] for b in batch]
    return images, questions, answers, raw_ans


# ─────────────────────────────────────────────────────────────────────────────
# B2 fine-tuning loop
# ─────────────────────────────────────────────────────────────────────────────

def finetune(args):
    device   = config.DEVICE
    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, "B2_blip_finetune")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = BLIPVQADataset(config.TRAIN_CSV, args.img_base, "train")
    val_ds   = BLIPVQADataset(config.VAL_CSV,   args.img_base, "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pil, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_pil, num_workers=2)

    model     = BLIPFineTuned(use_lora=config.USE_LORA, translate=args.translate).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")
    history  = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for images, questions, answers, _ in train_loader:
            batch = model.preprocess_batch(images, questions, answers)
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, questions, answers, _ in val_loader:
                batch    = model.preprocess_batch(images, questions, answers)
                out      = model(**batch)
                val_loss += out.loss.item()

        scheduler.step()
        tr_l = total_loss / len(train_loader)
        vl_l = val_loss   / len(val_loader)
        log  = {"epoch": epoch, "train_loss": tr_l, "val_loss": vl_l, "time": time.time()-t0}
        history.append(log)
        print(f"[B2] Epoch {epoch+1:03d} | train={tr_l:.4f} val={vl_l:.4f} | {log['time']:.1f}s")

        if vl_l < best_val:
            best_val = vl_l
            model.save_pretrained(os.path.join(ckpt_dir, "best"))
            print(f"  ✓ Best saved val_loss={best_val:.4f}")

    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[B2] Training complete. Best val_loss={best_val:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# B1 zero-shot evaluation
# ─────────────────────────────────────────────────────────────────────────────

def zero_shot_eval(args):
    from evaluate import compute_all_metrics
    device  = config.DEVICE
    test_ds = BLIPVQADataset(config.TEST_CSV, args.img_base, "test")
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate_pil, num_workers=2)

    model = BLIPZeroShot(translate=args.translate)
    all_preds, all_refs = [], []

    for images, questions, _, raw_ans in loader:
        preds = model.answer(images, questions)
        all_preds.extend(preds)
        all_refs.extend(raw_ans)

    metrics = compute_all_metrics(all_preds, all_refs)
    print("[B1 Zero-shot] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    out_path = os.path.join(config.LOG_DIR, "B1_zeroshot_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "samples": list(zip(all_refs[:20], all_preds[:20]))}, f,
                  ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["finetune", "zero_shot"], default="finetune")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--img_base",   type=str,   default=None)
    p.add_argument("--translate",  action="store_true",
                   help="Dịch vi→en trước khi vào BLIP (khuyến nghị nếu GPU nhỏ)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "finetune":
        finetune(args)
    else:
        zero_shot_eval(args)

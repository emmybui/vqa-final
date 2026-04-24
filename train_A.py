"""
Training script – Hướng A (A1: LSTM decoder, A2: Transformer decoder)
Usage:
    python train_A.py --decoder lstm       # A1
    python train_A.py --decoder transformer # A2
"""

import os, sys, argparse, time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(__file__))
import config
from data.dataset import VQAFoodDataset, Vocabulary, build_vocab_from_csvs
from models.model_A import build_model_A1, build_model_A2

# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder",    choices=["lstm", "transformer"], default="lstm")
    p.add_argument("--epochs",     type=int,   default=config.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=config.LR)
    p.add_argument("--img_base",   type=str,   default=None,
                   help="Override base dir cho image paths nếu CSV dùng đường dẫn tuyệt đối cũ")
    p.add_argument("--resume",     type=str,   default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Label-smoothing cross-entropy
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothCE(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.pad_idx   = pad_idx
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """logits (B*T, V), targets (B*T,)"""
        B, V = logits.shape
        log_prob = torch.log_softmax(logits, dim=-1)
        # smooth: (1-ε) * one_hot + ε/V
        with torch.no_grad():
            smooth_targets = torch.full_like(log_prob, self.smoothing / (V - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            smooth_targets[:, self.pad_idx] = 0
            mask = (targets == self.pad_idx)
            smooth_targets[mask] = 0
        loss = -(smooth_targets * log_prob).sum(dim=-1)
        return loss.sum() / (~mask).sum().float()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0

    for batch in loader:
        image  = batch["image"].to(device)
        q_ids  = batch["q_ids"].to(device)
        q_len  = batch["q_len"].to(device)
        a_in   = batch["a_in"].to(device)
        a_tgt  = batch["a_tgt"].to(device)

        optimizer.zero_grad()
        with autocast(enabled=(device == "cuda")):
            logits = model(image, q_ids, q_len, a_in)          # (B, T, V)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), a_tgt.reshape(B * T))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        # token-level accuracy (exclude pad)
        mask = (a_tgt != 0)
        preds = logits.argmax(-1)
        n_correct += (preds[mask] == a_tgt[mask]).sum().item()
        n_total   += mask.sum().item()

    return total_loss / len(loader), n_correct / max(n_total, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0

    for batch in loader:
        image  = batch["image"].to(device)
        q_ids  = batch["q_ids"].to(device)
        q_len  = batch["q_len"].to(device)
        a_in   = batch["a_in"].to(device)
        a_tgt  = batch["a_tgt"].to(device)

        logits = model(image, q_ids, q_len, a_in)
        B, T, V = logits.shape
        loss    = criterion(logits.reshape(B * T, V), a_tgt.reshape(B * T))
        total_loss += loss.item()

        mask = (a_tgt != 0)
        preds = logits.argmax(-1)
        n_correct += (preds[mask] == a_tgt[mask]).sum().item()
        n_total   += mask.sum().item()

    return total_loss / len(loader), n_correct / max(n_total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    device  = config.DEVICE
    tag     = f"A1_lstm" if args.decoder == "lstm" else "A2_transformer"
    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Vocabulary ──
    vocab_path_q = os.path.join(config.CHECKPOINT_DIR, "q_vocab.json")
    vocab_path_a = os.path.join(config.CHECKPOINT_DIR, "a_vocab.json")
    if os.path.exists(vocab_path_q):
        q_vocab = Vocabulary.load(vocab_path_q)
        a_vocab = Vocabulary.load(vocab_path_a)
    else:
        q_vocab, a_vocab = build_vocab_from_csvs(
            [config.TRAIN_CSV, config.VAL_CSV, config.TEST_CSV],
            config.CHECKPOINT_DIR,
        )

    # ── Datasets ──
    train_ds = VQAFoodDataset(config.TRAIN_CSV, q_vocab, a_vocab, split="train", img_base_dir=args.img_base)
    val_ds   = VQAFoodDataset(config.VAL_CSV,   q_vocab, a_vocab, split="val",   img_base_dir=args.img_base)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── Model ──
    build_fn = build_model_A1 if args.decoder == "lstm" else build_model_A2
    model    = build_fn(q_vocab_size=len(q_vocab), a_vocab_size=len(a_vocab)).to(device)
    print(f"[{tag}] Parameters: {model.count_parameters():,}")

    # ── Optimizer – discriminative LR ──
    cnn_params  = list(model.img_encoder.backbone.parameters())
    rest_params = [p for p in model.parameters() if not any(p is q for q in cnn_params)]
    optimizer   = AdamW([
        {"params": cnn_params,  "lr": config.LR_CNN},
        {"params": rest_params, "lr": args.lr},
    ], weight_decay=config.WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = LabelSmoothCE(len(a_vocab), pad_idx=a_vocab.word2idx[a_vocab.PAD])
    scaler    = GradScaler(enabled=(device == "cuda"))

    # ── Resume ──
    start_epoch = 0
    best_val_loss = float("inf")
    patience_cnt  = 0
    history = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Resume] epoch {start_epoch}")

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        log = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": vl_loss, "val_acc": vl_acc, "time": elapsed}
        history.append(log)
        print(f"[{tag}] Epoch {epoch+1:03d}/{args.epochs} | "
              f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val loss={vl_loss:.4f} acc={vl_acc:.4f} | {elapsed:.1f}s")

        # ── Save best ──
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_cnt  = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "q_vocab_len": len(q_vocab),
                "a_vocab_len": len(a_vocab),
                "decoder_type": args.decoder,
            }, os.path.join(ckpt_dir, "best.pt"))
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt"))

    # save history
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{tag}] Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()

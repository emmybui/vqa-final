"""
So sánh 4 cấu hình: A1 vs A2 vs B1 vs B2
Tạo bảng kết quả + biểu đồ
Usage: python compare.py
"""

import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))

import config
from data.dataset  import VQAFoodDataset, Vocabulary, build_vocab_from_csvs
from models.model_A import build_model_A1, build_model_A2
from evaluate       import compute_all_metrics, evaluate_model


# ─────────────────────────────────────────────────────────────────────────────

def load_A_model(ckpt_path, decoder_type, q_vocab, a_vocab):
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
    build = build_model_A1 if decoder_type == "lstm" else build_model_A2
    m = build(len(q_vocab), len(a_vocab)).to(config.DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m


def run_comparison(args):
    device = config.DEVICE

    # vocab
    q_vocab = Vocabulary.load(os.path.join(config.CHECKPOINT_DIR, "q_vocab.json"))
    a_vocab = Vocabulary.load(os.path.join(config.CHECKPOINT_DIR, "a_vocab.json"))

    # test loader (Hướng A)
    test_ds = VQAFoodDataset(config.TEST_CSV, q_vocab, a_vocab, split="test",
                              img_base_dir=args.img_base)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_results = {}

    # ── A1 ──
    a1_path = os.path.join(config.CHECKPOINT_DIR, "A1_lstm", "best.pt")
    if os.path.isfile(a1_path):
        print("\n[Compare] Evaluating A1 (LSTM decoder)...")
        m = load_A_model(a1_path, "lstm", q_vocab, a_vocab)
        all_results["A1_LSTM"] = evaluate_model(m, test_loader, a_vocab, device, "A1_LSTM")
    else:
        print(f"[Compare] A1 checkpoint not found: {a1_path}")

    # ── A2 ──
    a2_path = os.path.join(config.CHECKPOINT_DIR, "A2_transformer", "best.pt")
    if os.path.isfile(a2_path):
        print("\n[Compare] Evaluating A2 (Transformer decoder)...")
        m = load_A_model(a2_path, "transformer", q_vocab, a_vocab)
        all_results["A2_Transformer"] = evaluate_model(m, test_loader, a_vocab, device, "A2_Transformer")

    # ── B1 / B2 ──
    for tag, bpath in [("B1_ZeroShot", None), ("B2_FineTuned", os.path.join(config.CHECKPOINT_DIR, "B2_blip_finetune", "best"))]:
        b1_log = os.path.join(config.LOG_DIR, "B1_zeroshot_results.json")
        if tag == "B1_ZeroShot" and os.path.isfile(b1_log):
            with open(b1_log) as f:
                d = json.load(f)
            all_results["B1_ZeroShot"] = d["metrics"]
        if tag == "B2_FineTuned" and bpath and os.path.isdir(bpath):
            from models.model_B import BLIPFineTuned
            from train_B import BLIPVQADataset, collate_pil
            b_ds = BLIPVQADataset(config.TEST_CSV, args.img_base, "test")
            b_loader = DataLoader(b_ds, batch_size=8, shuffle=False, collate_fn=collate_pil)
            model_b = BLIPFineTuned.load_pretrained(bpath).to(device)
            model_b.eval()
            preds, refs = [], []
            with torch.no_grad():
                for images, questions, _, raw_ans in b_loader:
                    preds.extend(model_b.generate(images, questions))
                    refs.extend(raw_ans)
            all_results["B2_FineTuned"] = compute_all_metrics(preds, refs, use_bertscore=True)

    # ── Print comparison table ──
    if not all_results:
        print("[Compare] No models evaluated. Train first.")
        return

    METRICS = ["VQA_ExactMatch", "BLEU", "BLEU-1", "ROUGE_L", "METEOR", "BERTScore_F"]
    header = f"{'Metric':22s}" + "".join(f"{k:20s}" for k in all_results)
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(header)
    print("-"*80)
    for m in METRICS:
        row = f"{m:22s}"
        for cfg in all_results:
            v = all_results[cfg].get(m, 0.0)
            if isinstance(v, float):
                row += f"{v:20.4f}"
            else:
                row += f"{'N/A':20s}"
        print(row)
    print("="*80)

    # save
    out = os.path.join(config.LOG_DIR, "comparison.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved → {out}")

    # ── Plot ──
    try:
        _plot(all_results, METRICS[:5])
    except ImportError:
        print("[Plot] matplotlib not installed – skipping plot")


def _plot(all_results, metrics):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    configs = list(all_results.keys())
    x       = np.arange(len(metrics))
    width   = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, cfg in enumerate(configs):
        vals = [all_results[cfg].get(m, 0.0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=cfg)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("VQA Vietnamese Food – Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    out = os.path.join(config.LOG_DIR, "comparison_chart.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"[Plot] Saved → {out}")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_base",   type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run_comparison(parse_args())

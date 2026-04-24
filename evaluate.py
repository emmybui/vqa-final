"""
Evaluation Metrics cho VQA Vietnamese Food
  1. VQA Accuracy (exact match + soft)
  2. BLEU (1-4)
  3. ROUGE-L
  4. METEOR
  5. BERTScore
"""

import re, json, os
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Normalize: lowercase, remove punctuation."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\sàáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵýỷỹ]", "", s)
    return " ".join(s.split())


# ─────────────────────────────────────────────────────────────────────────────
# 1. VQA Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def vqa_exact_accuracy(preds: List[str], refs: List[str]) -> float:
    """Exact match (chuẩn hoá text)."""
    correct = sum(_normalize(p) == _normalize(r) for p, r in zip(preds, refs))
    return correct / len(preds)


def vqa_soft_accuracy(preds: List[str], refs_list: List[List[str]]) -> float:
    """
    VQA v2 soft accuracy: min(#humans agreeing / 3, 1)
    refs_list[i] = list of human answers for sample i (thường 10 answers)
    """
    scores = []
    for pred, refs in zip(preds, refs_list):
        p_norm = _normalize(pred)
        cnt    = sum(1 for r in refs if _normalize(r) == p_norm)
        scores.append(min(cnt / 3.0, 1.0))
    return np.mean(scores)


# ─────────────────────────────────────────────────────────────────────────────
# 2. BLEU
# ─────────────────────────────────────────────────────────────────────────────

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _bleu_n(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    pred_ng = _ngrams(pred_tokens, n)
    ref_ng  = _ngrams(ref_tokens,  n)
    clip    = sum(min(cnt, ref_ng[ng]) for ng, cnt in pred_ng.items())
    denom   = max(sum(pred_ng.values()), 1)
    return clip / denom


def corpus_bleu(preds: List[str], refs: List[str], max_n: int = 4) -> Dict[str, float]:
    import math
    results = {}
    for n in range(1, max_n + 1):
        precisions = [_bleu_n(p.split(), r.split(), n) for p, r in zip(preds, refs)]
        p = np.mean(precisions) if precisions else 0.0
        results[f"BLEU-{n}"] = p

    # BLEU-4 with brevity penalty
    pred_len = sum(len(p.split()) for p in preds)
    ref_len  = sum(len(r.split()) for r in refs)
    bp = 1.0 if pred_len >= ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    geom = np.exp(np.mean([np.log(max(results[f"BLEU-{n}"], 1e-10)) for n in range(1, max_n+1)]))
    results["BLEU"] = bp * geom
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROUGE-L
# ─────────────────────────────────────────────────────────────────────────────

def _lcs_length(a: List, b: List) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i%2][j] = dp[(i-1)%2][j-1] + 1
            else:
                dp[i%2][j] = max(dp[(i-1)%2][j], dp[i%2][j-1])
    return dp[m%2][n]


def rouge_l(pred: str, ref: str) -> float:
    p_tok, r_tok = pred.split(), ref.split()
    if not p_tok or not r_tok:
        return 0.0
    lcs = _lcs_length(p_tok, r_tok)
    precision = lcs / len(p_tok)
    recall    = lcs / len(r_tok)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def corpus_rouge_l(preds: List[str], refs: List[str]) -> float:
    return np.mean([rouge_l(_normalize(p), _normalize(r)) for p, r in zip(preds, refs)])


# ─────────────────────────────────────────────────────────────────────────────
# 4. METEOR (simplified, no stemming)
# ─────────────────────────────────────────────────────────────────────────────

def meteor_score(pred: str, ref: str, alpha: float = 0.9, gamma: float = 0.5) -> float:
    p_tok = set(pred.split())
    r_tok = ref.split()
    matches = sum(1 for t in r_tok if t in p_tok)
    if matches == 0:
        return 0.0
    prec = matches / max(len(pred.split()), 1)
    rec  = matches / max(len(r_tok), 1)
    f    = prec * rec / (alpha * prec + (1 - alpha) * rec)
    # chunk penalty
    chunks = 1  # simplified
    pen = gamma * (chunks / max(matches, 1)) ** 3
    return f * (1 - pen)


def corpus_meteor(preds: List[str], refs: List[str]) -> float:
    return np.mean([meteor_score(_normalize(p), _normalize(r)) for p, r in zip(preds, refs)])


# ─────────────────────────────────────────────────────────────────────────────
# 5. BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def compute_bertscore(preds: List[str], refs: List[str]) -> Dict[str, float]:
    try:
        from bert_score import score as bs_score
        P, R, F = bs_score(preds, refs, lang="vi", verbose=False, rescale_with_baseline=True)
        return {
            "BERTScore_P": P.mean().item(),
            "BERTScore_R": R.mean().item(),
            "BERTScore_F": F.mean().item(),
        }
    except ImportError:
        print("[BERTScore] bert-score not installed. pip install bert-score")
        return {"BERTScore_F": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per question-type breakdown
# ─────────────────────────────────────────────────────────────────────────────

def per_type_accuracy(preds: List[str], refs: List[str], qtypes: List[str]) -> Dict[str, float]:
    type_scores: Dict[str, List[float]] = {}
    for p, r, t in zip(preds, refs, qtypes):
        score = 1.0 if _normalize(p) == _normalize(r) else 0.0
        type_scores.setdefault(t, []).append(score)
    return {t: np.mean(v) for t, v in sorted(type_scores.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Master function
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    preds: List[str],
    refs:  List[str],
    qtypes: List[str] = None,
    use_bertscore: bool = True,
) -> Dict[str, float]:
    metrics = {}
    metrics["VQA_ExactMatch"] = vqa_exact_accuracy(preds, refs)
    metrics.update(corpus_bleu(preds, refs))
    metrics["ROUGE_L"]  = corpus_rouge_l(preds, refs)
    metrics["METEOR"]   = corpus_meteor(preds, refs)
    if use_bertscore:
        metrics.update(compute_bertscore(preds, refs))
    if qtypes:
        metrics["per_type"] = per_type_accuracy(preds, refs, qtypes)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate a saved model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, loader, a_vocab, device, tag: str = "") -> Dict:
    """Run generation on test set and compute metrics."""
    import torch
    from data.dataset import tokenize_vi

    bos = a_vocab.word2idx[a_vocab.BOS]
    eos = a_vocab.word2idx[a_vocab.EOS]
    preds_str, refs_str, qtypes = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            image  = batch["image"].to(device)
            q_ids  = batch["q_ids"].to(device)
            q_len  = batch["q_len"].to(device)
            gen    = model.generate(image, q_ids, q_len, bos, eos)  # (B, L)
            for ids in gen:
                preds_str.append(a_vocab.decode(ids.tolist()))
            refs_str.extend(batch["raw_answer"])
            qtypes.extend(batch["question_type"])

    metrics = compute_all_metrics(preds_str, refs_str, qtypes)
    print(f"\n{'='*50}")
    print(f"[Eval] {tag}")
    for k, v in metrics.items():
        if k != "per_type":
            print(f"  {k:20s}: {v:.4f}")
    if "per_type" in metrics:
        print("  Per question type:")
        for t, v in metrics["per_type"].items():
            print(f"    {t:25s}: {v:.4f}")

    out = os.path.join(config.LOG_DIR, f"{tag}_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        # Convert per_type for JSON serialization
        m_save = {k: (v if k != "per_type" else v) for k, v in metrics.items()}
        json.dump({"metrics": m_save, "samples": list(zip(refs_str[:30], preds_str[:30]))},
                  f, ensure_ascii=False, indent=2)
    print(f"  Saved → {out}")
    return metrics


if __name__ == "__main__":
    # Quick sanity check
    preds = ["đây là món bánh canh", "có", "màu trắng"]
    refs  = ["đây là món bánh canh",  "có", "màu vàng"]
    m     = compute_all_metrics(preds, refs, use_bertscore=False)
    for k, v in m.items():
        if k != "per_type":
            print(f"{k}: {v:.4f}")

# 🍜 VQA Vietnamese Food – Hệ Thống Hỏi Đáp Hình Ảnh Món Ăn Việt Nam

Dự án cuối kỳ môn Học Sâu – VQA (Visual Question Answering) tiếng Việt  
Domain: **Món ăn Việt Nam** | 4 cấu hình bắt buộc: A1, A2, B1, B2

---

## 🏗️ Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                     VQA Vietnamese Food                      │
│                                                             │
│  Input: [Image] + [Câu hỏi tiếng Việt]                     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              HƯỚNG A – Kiến trúc rời                  │  │
│  │                                                       │  │
│  │  [Image] → EfficientNet-B4 → Region Features (49×D)  │  │
│  │                          ↘                           │  │
│  │                     Co-Attention  → Fused (D×2)       │  │
│  │                          ↗                           │  │
│  │  [Text]  → Embedding → BiLSTM   → Token Features     │  │
│  │                                                       │  │
│  │  A1: Fused → LSTM Decoder → Answer                   │  │
│  │  A2: Fused → Transformer Decoder → Answer            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              HƯỚNG B – Multimodal Pretrained          │  │
│  │                                                       │  │
│  │  B1: BLIP-VQA-base (zero-shot) + vi→en translation   │  │
│  │  B2: BLIP-VQA-base fine-tuned + LoRA                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
vqa_food/
├── config.py              # Tất cả hyperparameters
├── requirements.txt
├── data/
│   └── dataset.py         # VQAFoodDataset, Vocabulary, Tokenizer
├── models/
│   ├── image_encoder.py   # EfficientNet-B4 / ResNet50 / ViT
│   ├── text_encoder.py    # BiLSTM encoder
│   ├── fusion.py          # Co-Attention + Concat fusion
│   ├── decoder.py         # LSTMDecoder + TransformerDecoder
│   ├── model_A.py         # VQAModel (A1 + A2)
│   └── model_B.py         # BLIP fine-tune (B1 + B2)
├── train_A.py             # Train A1 / A2
├── train_B.py             # Train B2 / eval B1
├── evaluate.py            # VQA Acc, BLEU, ROUGE-L, METEOR, BERTScore
├── compare.py             # So sánh 4 cấu hình + biểu đồ
├── demo.py                # Gradio demo
└── Dataset/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── Train/
    ├── Validate/
    └── Test/
```

---

## 🚀 Hướng dẫn chạy

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dataset

Nếu `image_path` dùng đường dẫn tuyệt đối cũ (trên máy khác), dùng `--img_base`:
```bash
# img_base = thư mục chứa ảnh mới
python train_A.py --img_base /path/to/Dataset
```

### 3. Train Hướng A

```bash
# A1 – LSTM Decoder
python train_A.py --decoder lstm --epochs 50 --batch_size 32

# A2 – Transformer Decoder
python train_A.py --decoder transformer --epochs 50 --batch_size 32
```

### 4. Train Hướng B

```bash
# B1 – Zero-shot evaluation (không cần train)
python train_B.py --mode zero_shot --translate

# B2 – Fine-tune BLIP + LoRA
python train_B.py --mode finetune --epochs 10 --batch_size 8 --translate
```

### 5. So sánh kết quả

```bash
python compare.py
```

Xuất ra `logs/comparison.json` và `logs/comparison_chart.png`

### 6. Demo Gradio

```bash
python demo.py \
  --model_A1 checkpoints/A1_lstm/best.pt \
  --model_A2 checkpoints/A2_transformer/best.pt \
  --model_B2 checkpoints/B2_blip_finetune/best \
  --share
```

---

## 🧠 Giải thích kiến trúc

### Image Encoder – EfficientNet-B4 or ResNet 101 (can changes later)
- Pretrained trên ImageNet
- Lấy spatial features: **(B, 49, 512)** grid 7×7
- Data augmentation: flip, rotate, crop, color jitter

### Text Encoder – BiLSTM
- Embedding dim: 256 → BiLSTM hidden: 512 (256 mỗi chiều)
- Output: **(B, T_q, 512)** token features
- Tiếng Việt: dùng `underthesea.word_tokenize` nếu có

### Fusion – Co-Attention
- **Image-guided text**: text tokens attend to image regions
- **Text-guided image**: image regions attend to text tokens  
- Output: concat mean-pooled → **(B, 1024)**

### A1 – LSTM Decoder
- h₀ khởi tạo từ context vector
- Mỗi bước: [embedding; context] → LSTM → Linear
- Teacher forcing khi training, greedy khi inference

### A2 – Transformer Decoder
- Pre-LN Transformer (ổn định hơn khi training)
- Context → memory (B, 1, D) cho cross-attention
- Causal masking đảm bảo autoregressive
- Positional encoding sinusoidal

### B – BLIP Fine-tuning
- Base: `Salesforce/blip-vqa-base` (~400MB, chạy được trên 8GB GPU)
- LoRA rank=16 giảm ~90% trainable params
- Chiến lược tiếng Việt: dịch vi→en qua `Helsinki-NLP/opus-mt-vi-en`

---

## 📊 Metrics đánh giá

| Metric | Mô tả |
|--------|-------|
| VQA Exact Match | Chuẩn hoá → so khớp chính xác |
| BLEU-1/2/3/4 | N-gram precision |
| ROUGE-L | Longest common subsequence |
| METEOR | Unigram matching với penalty |
| BERTScore-F | Semantic similarity (PhoBERT) |

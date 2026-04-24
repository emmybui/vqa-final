"""
Configuration cho toàn bộ project VQA Vietnamese Food
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "Dataset")
TRAIN_CSV     = os.path.join(DATASET_DIR, "train.csv")
VAL_CSV       = os.path.join(DATASET_DIR, "val.csv")
TEST_CSV      = os.path.join(DATASET_DIR, "test.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── Image ──────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224          # input size cho CNN
IMG_MEAN    = [0.485, 0.456, 0.406]
IMG_STD     = [0.229, 0.224, 0.225]
IMG_ENCODER = "efficientnet_b4"   # hoặc "resnet50", "vit_b_16"

# ── Text ───────────────────────────────────────────────────────────────────────
MAX_Q_LEN   = 32           # max token câu hỏi
MAX_A_LEN   = 20           # max token câu trả lời
VOCAB_SIZE  = 10000        # kích thước vocabulary
EMBED_DIM   = 256          # word embedding dimension
HIDDEN_DIM  = 512          # BiLSTM hidden size (mỗi chiều = 256)
NUM_LAYERS  = 2            # số lớp BiLSTM

# ── Fusion / Attention ─────────────────────────────────────────────────────────
FUSION_DIM      = 512      # chiều sau khi fuse
ATTENTION_HEADS = 8        # số head attention
DROPOUT         = 0.3

# ── Transformer Decoder (A2) ───────────────────────────────────────────────────
TRANS_LAYERS    = 4
TRANS_FFN_DIM   = 1024

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
NUM_EPOCHS   = 50
LR           = 1e-4
LR_CNN       = 1e-5        # lower LR cho pretrained CNN
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
PATIENCE     = 8           # early stopping patience
LABEL_SMOOTH = 0.1

# ── Direction B (BLIP-2) ───────────────────────────────────────────────────────
BLIP_MODEL_NAME = "Salesforce/blip2-opt-2.7b"    # hoặc "Salesforce/blip-vqa-base" (nhẹ hơn)
BLIP_LITE       = "Salesforce/blip-vqa-base"     # model nhẹ để chạy thử
USE_LORA        = True
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_BEAM_SIZE  = 4

# ── Device ─────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Config] Device: {DEVICE}")

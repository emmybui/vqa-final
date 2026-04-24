"""
Hướng B – Multimodal Pretrained: BLIP / BLIP-2
  B1: Zero-shot (không fine-tune)
  B2: Fine-tuned (LoRA nếu cần)

Chiến lược tiếng Việt:
  - Dùng trực tiếp: model hiểu tiếng Việt cơ bản (BLIP-2 với LLM backbone lớn)
  - Hoặc dịch: vi→en trước khi vào model (dùng M2M-100 nhẹ)
"""

import os, torch
import torch.nn as nn
from typing import List
import config

# ─────────────────────────────────────────────────────────────────────────────
# Lazy import – tránh lỗi nếu chưa cài thư viện
# ─────────────────────────────────────────────────────────────────────────────

def _load_blip_base():
    """Tải BLIP-base (nhẹ ~400MB), phù hợp cho cả B1 & B2."""
    from transformers import BlipProcessor, BlipForQuestionAnswering
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model     = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model


def _load_blip2():
    """Tải BLIP-2 (nặng hơn, cần GPU ≥16GB hoặc 8bit)."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained(config.BLIP_LITE)
    model     = Blip2ForConditionalGeneration.from_pretrained(
        config.BLIP_LITE,
        load_in_8bit=True,
        device_map="auto",
    )
    return processor, model


# ─────────────────────────────────────────────────────────────────────────────
# Optional: dịch câu hỏi vi→en
# ─────────────────────────────────────────────────────────────────────────────

class ViEnTranslator:
    """Dịch nhanh vi→en bằng Helsinki-NLP/opus-mt-vi-en."""

    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        model_name = "Helsinki-NLP/opus-mt-vi-en"
        self.tok   = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def translate(self, texts: List[str]) -> List[str]:
        inputs = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        out    = self.model.generate(**inputs, num_beams=4, max_new_tokens=128)
        return self.tok.batch_decode(out, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# B1 – Zero-shot wrapper
# ─────────────────────────────────────────────────────────────────────────────

class BLIPZeroShot:
    """B1: zero-shot inference, không fine-tune."""

    def __init__(self, translate: bool = True):
        print("[BLIP B1] Loading BLIP-VQA-base...")
        self.processor, self.model = _load_blip_base()
        self.model.eval().to(config.DEVICE)
        self.translator = ViEnTranslator() if translate else None
        self.translate  = translate

    def answer(self, images, questions: List[str]) -> List[str]:
        """
        images    : list of PIL.Image
        questions : list of str (tiếng Việt)
        """
        if self.translate and self.translator:
            questions = self.translator.translate(questions)

        inputs = self.processor(images=images, text=questions,
                                return_tensors="pt", padding=True).to(config.DEVICE)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.batch_decode(out, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# B2 – Fine-tuned BLIP + LoRA
# ─────────────────────────────────────────────────────────────────────────────

class BLIPFineTuned(nn.Module):
    """B2: Fine-tune BLIP-VQA-base với LoRA (PEFT)."""

    def __init__(self, use_lora: bool = config.USE_LORA, translate: bool = True):
        super().__init__()
        from transformers import BlipProcessor, BlipForQuestionAnswering

        self.processor  = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.blip       = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        self.translate  = translate
        self.translator = ViEnTranslator() if translate else None

        if use_lora:
            self._apply_lora()

    def _apply_lora(self):
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_cfg = LoraConfig(
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
                # target all linear layers in language model
                target_modules=["query", "key", "value", "dense"],
            )
            self.blip = get_peft_model(self.blip, lora_cfg)
            self.blip.print_trainable_parameters()
        except ImportError:
            print("[BLIP B2] peft not installed – fine-tuning all params")

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        """Direct call to BLIP internal forward."""
        return self.blip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def preprocess_batch(self, images, questions: List[str], answers: List[str] = None):
        """Tiền xử lý batch cho training / inference."""
        if self.translate and self.translator:
            questions = self.translator.translate(questions)
            if answers:
                answers = self.translator.translate(answers)

        if answers:
            encoding = self.processor(
                images=images, text=questions,
                text_pair=answers,
                return_tensors="pt", padding=True, truncation=True,
            )
            # labels = token ids of answers
            ans_enc = self.processor.tokenizer(
                answers, return_tensors="pt", padding=True, truncation=True, max_length=config.MAX_A_LEN
            )
            encoding["labels"] = ans_enc["input_ids"]
        else:
            encoding = self.processor(
                images=images, text=questions,
                return_tensors="pt", padding=True, truncation=True,
            )
        return {k: v.to(config.DEVICE) for k, v in encoding.items()}

    @torch.no_grad()
    def generate(self, images, questions: List[str]) -> List[str]:
        if self.translate and self.translator:
            questions = self.translator.translate(questions)
        inputs = self.processor(images=images, text=questions,
                                return_tensors="pt", padding=True).to(config.DEVICE)
        out = self.blip.generate(**inputs, max_new_tokens=30)
        return self.processor.batch_decode(out, skip_special_tokens=True)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.blip.save_pretrained(path)
        self.processor.save_pretrained(path)

    @classmethod
    def load_pretrained(cls, path: str, use_lora: bool = False) -> "BLIPFineTuned":
        from transformers import BlipProcessor, BlipForQuestionAnswering
        obj = cls.__new__(cls)
        super(BLIPFineTuned, obj).__init__()
        obj.processor  = BlipProcessor.from_pretrained(path)
        obj.blip       = BlipForQuestionAnswering.from_pretrained(path)
        obj.translate  = False
        obj.translator = None
        return obj


if __name__ == "__main__":
    print("BLIP models defined. Run train_B.py to train.")

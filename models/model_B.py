import os
import torch
import torch.nn as nn
from typing import List, Optional
import config


class ViEnTranslator:
    """Dịch nhanh bằng Helsinki-NLP/opus-mt-vi-en."""

    def __init__(self):
        from transformers import MarianMTModel, MarianTokenizer
        model_name = "Helsinki-NLP/opus-mt-vi-en"
        self.tok   = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def translate(self, texts: List[str]) -> List[str]:
        inputs = self.tok(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        )
        out = self.model.generate(**inputs, num_beams=4, max_new_tokens=128)
        return self.tok.batch_decode(out, skip_special_tokens=True)


def _load_blip_base():
    """Tải BLIP-VQA-base (~400MB). Phù hợp cả B1 và B2."""
    from transformers import BlipProcessor, BlipForQuestionAnswering
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model     = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model


class BLIPZeroShot:
    """B1: zero-shot inference, không fine-tune."""

    def __init__(self, translate: bool = True):
        print("[BLIP B1] Loading BLIP-VQA-base...")
        self.processor, self.model = _load_blip_base()
        self.model.eval().to(config.DEVICE)
        self.translator = ViEnTranslator() if translate else None
        self.translate  = translate

    def answer(self, images, questions: List[str]) -> List[str]:
        if self.translate and self.translator:
            questions = self.translator.translate(questions)

        inputs = self.processor(
            images=images, text=questions,
            return_tensors="pt", padding=True,
        ).to(config.DEVICE)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.batch_decode(out, skip_special_tokens=True)


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
            from peft import get_peft_model, LoraConfig
            lora_cfg = LoraConfig(
                r            = config.LORA_R,
                lora_alpha   = config.LORA_ALPHA,
                lora_dropout = config.LORA_DROPOUT,
                bias         = "none",
                target_modules = ["query", "key", "value", "dense"],
            )
            self.blip = get_peft_model(self.blip, lora_cfg)
            self.blip.print_trainable_parameters()
        except ImportError:
            print("[BLIP B2] peft not installed – fine-tuning all params")

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        return self.blip(
            pixel_values   = pixel_values,
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )

    def preprocess_batch(
        self,
        images,
        questions: List[str],
        answers: Optional[List[str]] = None,
    ) -> dict:
        if self.translate and self.translator:
            questions = self.translator.translate(questions)
            if answers:
                answers = self.translator.translate(answers)

        # Encode ảnh + câu hỏi
        encoding = self.processor(
            images=images,
            text=questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if answers is not None:
            # FIX: Encode answers riêng bằng tokenizer, gán làm labels
            ans_enc = self.processor.tokenizer(
                answers,
                return_tensors = "pt",
                padding        = True,
                truncation     = True,
                max_length     = config.MAX_A_LEN,
            )
            labels = ans_enc["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            encoding["labels"] = labels

        return {k: v.to(config.DEVICE) for k, v in encoding.items()}

    @torch.no_grad()
    def generate(self, images, questions: List[str]) -> List[str]:
        if self.translate and self.translator:
            questions = self.translator.translate(questions)
        inputs = self.processor(
            images=images, text=questions,
            return_tensors="pt", padding=True,
        ).to(config.DEVICE)
        out = self.blip.generate(**inputs, max_new_tokens=30)
        return self.processor.batch_decode(out, skip_special_tokens=True)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.blip.save_pretrained(path)
        self.processor.save_pretrained(path)

    @classmethod
    def load_pretrained(cls, path: str) -> "BLIPFineTuned":
        from transformers import BlipProcessor, BlipForQuestionAnswering
        obj            = cls.__new__(cls)
        super(BLIPFineTuned, obj).__init__()
        obj.processor  = BlipProcessor.from_pretrained(path)
        obj.blip       = BlipForQuestionAnswering.from_pretrained(path)
        obj.translate  = False
        obj.translator = None
        return obj


if __name__ == "__main__":
    print("BLIP models defined. Run train_B.py to train.")

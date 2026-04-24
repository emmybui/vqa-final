"""
Inference + Gradio Demo
Usage:
    python demo.py --model_A1 checkpoints/A1_lstm/best.pt
                   --model_A2 checkpoints/A2_transformer/best.pt
                   --model_B2 checkpoints/B2_blip_finetune/best
"""

import os, sys, argparse
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))

import config
from data.dataset  import Vocabulary, tokenize_vi, get_transforms
from models.model_A import VQAModel, build_model_A1, build_model_A2
from models.model_B import BLIPFineTuned


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_model_A(ckpt_path: str, decoder_type: str) -> tuple:
    ckpt    = torch.load(ckpt_path, map_location=config.DEVICE)
    q_vocab = Vocabulary.load(os.path.join(config.CHECKPOINT_DIR, "q_vocab.json"))
    a_vocab = Vocabulary.load(os.path.join(config.CHECKPOINT_DIR, "a_vocab.json"))
    build   = build_model_A1 if decoder_type == "lstm" else build_model_A2
    model   = build(len(q_vocab), len(a_vocab)).to(config.DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, q_vocab, a_vocab


def load_model_B(ckpt_path: str) -> BLIPFineTuned:
    return BLIPFineTuned.load_pretrained(ckpt_path).to(config.DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Single inference helpers
# ─────────────────────────────────────────────────────────────────────────────

_transform = get_transforms("val")


@torch.no_grad()
def predict_A(model: VQAModel, q_vocab: Vocabulary, a_vocab: Vocabulary,
              image: Image.Image, question: str, beam: int = 1) -> str:
    img_t = _transform(image.convert("RGB")).unsqueeze(0).to(config.DEVICE)
    q_tok = tokenize_vi(question)
    q_ids = torch.LongTensor(q_vocab.encode(q_tok, config.MAX_Q_LEN)).unsqueeze(0).to(config.DEVICE)
    q_len = torch.LongTensor([[min(len(q_tok), config.MAX_Q_LEN)]]).to(config.DEVICE)

    bos = a_vocab.word2idx[a_vocab.BOS]
    eos = a_vocab.word2idx[a_vocab.EOS]
    gen = model.generate(img_t, q_ids, q_len, bos, eos, beam_size=beam)  # (1, L)
    return a_vocab.decode(gen[0].tolist())


@torch.no_grad()
def predict_B(model: BLIPFineTuned, image: Image.Image, question: str) -> str:
    answers = model.generate([image], [question])
    return answers[0]


# ─────────────────────────────────────────────────────────────────────────────
# Gradio Demo
# ─────────────────────────────────────────────────────────────────────────────

def build_demo(models_dict: dict):
    """
    models_dict = {
        "A1 (LSTM)":        (model_a1, q_vocab, a_vocab),
        "A2 (Transformer)": (model_a2, q_vocab, a_vocab),
        "B2 (BLIP)":        model_b2,
    }
    """
    import gradio as gr

    def predict(image, question, model_choice):
        if image is None or not question.strip():
            return "Vui lòng cung cấp ảnh và câu hỏi."
        pil_img = Image.fromarray(image) if not isinstance(image, Image.Image) else image

        if model_choice in ("A1 (LSTM)", "A2 (Transformer)"):
            m, qv, av = models_dict[model_choice]
            return predict_A(m, qv, av, pil_img, question)
        else:
            return predict_B(models_dict["B2 (BLIP)"], pil_img, question)

    # sample questions
    EXAMPLES = [
        ["Dataset/Test/Banh_mi/01.jpg", "Đây là món gì?",                  "A1 (LSTM)"],
        ["Dataset/Test/Banh_mi/01.jpg", "Đây có phải món phở không?",       "A2 (Transformer)"],
        ["Dataset/Test/Banh_mi/01.jpg", "Món này thường ăn nóng hay nguội?","B2 (BLIP)"],
    ]

    with gr.Blocks(title="🍜 VQA Món Ăn Việt Nam") as demo:
        gr.Markdown("# 🍜 Hỏi Đáp Hình Ảnh – Món Ăn Việt Nam")
        gr.Markdown("Upload ảnh món ăn và đặt câu hỏi tiếng Việt!")

        with gr.Row():
            with gr.Column(scale=1):
                inp_img      = gr.Image(label="Ảnh món ăn", type="pil")
                inp_question = gr.Textbox(label="Câu hỏi tiếng Việt", placeholder="Đây là món gì?")
                inp_model    = gr.Radio(
                    choices=list(models_dict.keys()),
                    value=list(models_dict.keys())[0],
                    label="Chọn mô hình",
                )
                btn = gr.Button("Trả lời 🔍", variant="primary")

            with gr.Column(scale=1):
                out_answer = gr.Textbox(label="Câu trả lời", lines=3, interactive=False)

        btn.click(predict, inputs=[inp_img, inp_question, inp_model], outputs=out_answer)

        gr.Examples(
            examples=EXAMPLES,
            inputs=[inp_img, inp_question, inp_model],
            outputs=out_answer,
            fn=predict,
            cache_examples=False,
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_A1", default=None)
    p.add_argument("--model_A2", default=None)
    p.add_argument("--model_B2", default=None)
    p.add_argument("--port",     type=int, default=7860)
    p.add_argument("--share",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models_dict = {}

    if args.model_A1 and os.path.isfile(args.model_A1):
        print("[Demo] Loading A1...")
        m, qv, av = load_model_A(args.model_A1, "lstm")
        models_dict["A1 (LSTM)"] = (m, qv, av)

    if args.model_A2 and os.path.isfile(args.model_A2):
        print("[Demo] Loading A2...")
        m, qv, av = load_model_A(args.model_A2, "transformer")
        models_dict["A2 (Transformer)"] = (m, qv, av)

    if args.model_B2 and os.path.isdir(args.model_B2):
        print("[Demo] Loading B2...")
        models_dict["B2 (BLIP)"] = load_model_B(args.model_B2)

    if not models_dict:
        # demo mode: giả lập nếu chưa có checkpoint
        print("[Demo] No checkpoints found – running in placeholder mode")
        models_dict["A1 (LSTM)"] = None

    demo = build_demo(models_dict)
    demo.launch(server_port=args.port, share=args.share)

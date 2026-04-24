import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import config

class BiLSTMEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int  = config.EMBED_DIM,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        out_dim: int    = config.FUSION_DIM,
        pad_idx: int    = 0,
        pretrained_emb: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)

        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = config.DROPOUT if num_layers > 1 else 0.0,
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
        )
        self.out_dim = out_dim

    def forward(self, q_ids: torch.Tensor, q_len: torch.Tensor):
        """
        q_ids : (B, T)  LongTensor
        q_len : (B, 1)  actual lengths
        Returns:
            outputs : (B, T, out_dim)
            pooled  : (B, out_dim)
        """
        emb = self.embedding(q_ids)
        emb = F.dropout(emb, p=config.DROPOUT, training=self.training)

        lengths = q_len.squeeze(1).cpu().clamp(min=1)
        packed  = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, (hn, _) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=q_ids.size(1)
        )
        # outputs: (B, T, hidden*2)

        # concat last forward + last backward hidden của top layer
        fwd_last = hn[-2]  # (B, hidden)
        bwd_last = hn[-1]  # (B, hidden)
        pooled   = torch.cat([fwd_last, bwd_last], dim=-1)  # (B, hidden*2)

        outputs = self.proj(outputs)  # (B, T, out_dim)
        pooled  = self.proj(pooled)   # (B, out_dim)
        return outputs, pooled



class PhoBERTEncoder(nn.Module):

    def __init__(
        self,
        out_dim: int      = config.FUSION_DIM,
        freeze_layers: int = 8,   # đóng băng 8/12 layer đầu để tiết kiệm VRAM
        model_name: str   = "vinai/phobert-base-v2",
    ):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.out_dim = out_dim

        # đóng băng n layer đầu
        if freeze_layers > 0:
            modules_to_freeze = [
                self.bert.embeddings,
                *self.bert.encoder.layer[:freeze_layers],
            ]
            for m in modules_to_freeze:
                for p in m.parameters():
                    p.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.Linear(768, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.proj(out.last_hidden_state)  # (B, T, out_dim)
        pooled  = self.proj(out.last_hidden_state[:, 0])  # [CLS] token
        return outputs, pooled


def build_text_encoder(encoder_type: str = config.TXT_ENCODER, vocab_size: int = None, **kw):
    if encoder_type == "bilstm":
        assert vocab_size is not None, "vocab_size bắt buộc khi dùng BiLSTM"
        return BiLSTMEncoder(vocab_size=vocab_size, **kw)
    elif encoder_type == "phobert":
        return PhoBERTEncoder(**kw)
    else:
        raise ValueError(f"Unknown text encoder: {encoder_type}")


if __name__ == "__main__":
    # test BiLSTM
    enc = BiLSTMEncoder(vocab_size=5000).to(config.DEVICE)
    ids  = torch.randint(0, 5000, (4, config.MAX_Q_LEN)).to(config.DEVICE)
    lens = torch.randint(5, config.MAX_Q_LEN, (4, 1)).to(config.DEVICE)
    o, p = enc(ids, lens)
    print("BiLSTM — outputs:", o.shape, "pooled:", p.shape)

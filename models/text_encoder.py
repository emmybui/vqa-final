"""
Text Encoder – BiLSTM
Output: (B, MAX_Q_LEN, HIDDEN_DIM*2) + (B, HIDDEN_DIM*2) pooled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import config


class BiLSTMEncoder(nn.Module):
    """
    Encode câu hỏi bằng BiLSTM.
    Output:
        outputs : (B, T, hidden*2)   — all hidden states (cho co-attention)
        pooled  : (B, hidden*2)      — final representation
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int   = config.EMBED_DIM,
        hidden_dim: int  = config.HIDDEN_DIM,
        num_layers: int  = config.NUM_LAYERS,
        out_dim: int     = config.FUSION_DIM,
        pad_idx: int     = 0,
        pretrained_emb: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = config.DROPOUT if num_layers > 1 else 0.0,
        )

        # projection → FUSION_DIM
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
        emb = self.embedding(q_ids)                    # (B, T, E)
        emb = F.dropout(emb, p=config.DROPOUT, training=self.training)

        # pack for efficiency
        lengths = q_len.squeeze(1).cpu().clamp(min=1)
        packed  = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, (hn, _) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True,
                                                       total_length=q_ids.size(1))
        # outputs: (B, T, hidden*2)

        # pooled: concat last forward + last backward hidden of top layer
        # hn: (num_layers*2, B, hidden)
        fwd_last = hn[-2]   # (B, hidden)
        bwd_last = hn[-1]   # (B, hidden)
        pooled   = torch.cat([fwd_last, bwd_last], dim=-1)   # (B, hidden*2)

        outputs = self.proj(outputs)       # (B, T, out_dim)
        pooled  = self.proj(pooled)        # (B, out_dim)
        return outputs, pooled


if __name__ == "__main__":
    enc = BiLSTMEncoder(vocab_size=5000).to(config.DEVICE)
    ids = torch.randint(0, 5000, (4, config.MAX_Q_LEN)).to(config.DEVICE)
    lens = torch.randint(5, config.MAX_Q_LEN, (4, 1)).to(config.DEVICE)
    o, p = enc(ids, lens)
    print("outputs:", o.shape, "pooled:", p.shape)
    # Expected: outputs (4, 32, 512)  pooled (4, 512)

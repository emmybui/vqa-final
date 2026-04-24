import math
import torch
import torch.nn as nn
import config


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int   = config.EMBED_DIM,
        hidden_dim: int  = config.HIDDEN_DIM,
        context_dim: int = config.FUSION_DIM * 2,
        num_layers: int  = 2,
        pad_idx: int     = 0,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.context_dim = context_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.ctx2h = nn.Linear(context_dim, hidden_dim * num_layers)
        self.ctx2c = nn.Linear(context_dim, hidden_dim * num_layers)

        self.lstm = nn.LSTM(
            input_size  = embed_dim + context_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = config.DROPOUT if num_layers > 1 else 0.0,
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, vocab_size),
        )

    def _init_hidden(self, context):
        B = context.size(0)
        h = torch.tanh(self.ctx2h(context))
        c = torch.tanh(self.ctx2c(context))
        h = h.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = c.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        return h, c

    def forward(self, context: torch.Tensor, a_in: torch.Tensor):
        B, T = a_in.shape
        h, c = self._init_hidden(context)

        emb     = self.embedding(a_in)                        # (B, T, E)
        ctx_exp = context.unsqueeze(1).expand(-1, T, -1)      # (B, T, ctx_dim)
        lstm_in = torch.cat([emb, ctx_exp], dim=-1)

        out, _  = self.lstm(lstm_in, (h, c))                  # (B, T, hidden)
        logits  = self.out(torch.cat([out, ctx_exp], dim=-1)) # (B, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, context: torch.Tensor, bos_idx: int, eos_idx: int,
                 max_len: int = config.MAX_A_LEN):
        """Greedy decoding. LSTM chỉ hỗ trợ greedy."""
        B       = context.size(0)
        h, c    = self._init_hidden(context)
        token   = torch.full((B, 1), bos_idx, dtype=torch.long, device=context.device)
        generated = []
        for _ in range(max_len):
            emb     = self.embedding(token)
            ctx_exp = context.unsqueeze(1)
            lstm_in = torch.cat([emb, ctx_exp], dim=-1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            logit = self.out(torch.cat([out, ctx_exp], dim=-1))
            token = logit.argmax(dim=-1)
            generated.append(token)
            if (token == eos_idx).all():
                break
        return torch.cat(generated, dim=1)  # (B, L)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = config.DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerAnswerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int  = config.FUSION_DIM,
        num_layers: int = config.TRANS_LAYERS,
        num_heads: int  = config.ATTENTION_HEADS,
        ffn_dim: int    = config.TRANS_FFN_DIM,
        pad_idx: int    = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_idx   = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(embed_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = ffn_dim,
            dropout         = config.DROPOUT,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,   # Pre-LN for stability
        )
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        # FIX: Weight tying embedding ↔ output projection
        # Giảm ~vocab_size × embed_dim params, cải thiện generalization
        self.out_proj.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        # out_proj.weight đã tie với embedding, không cần init riêng

    def _make_causal_mask(self, T: int, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(
        self,
        memory: torch.Tensor,
        a_in: torch.Tensor,
        memory_key_padding_mask: torch.Tensor = None,   # (B, Nmem) True=pad
    ):
        """
        memory : (B, Nmem, embed_dim)  — img_feat + txt_feat concatenated
        a_in   : (B, T)
        Returns logits (B, T, vocab_size)
        """
        T      = a_in.size(1)
        tgt    = self.pos_enc(self.embedding(a_in) * math.sqrt(self.embed_dim))
        causal = self._make_causal_mask(T, a_in.device)
        tgt_key_pad = (a_in == self.pad_idx)

        out = self.decoder(
            tgt, memory,
            tgt_mask                = causal,
            tgt_key_padding_mask    = tgt_key_pad,
            memory_key_padding_mask = memory_key_padding_mask,
        )                                      # (B, T, embed_dim)
        return self.out_proj(out)              # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int  = config.MAX_A_LEN,
        beam_size: int = 1,
        memory_key_padding_mask: torch.Tensor = None,
    ):
        if beam_size == 1:
            return self._greedy(memory, bos_idx, eos_idx, max_len, memory_key_padding_mask)
        return self._beam(memory, bos_idx, eos_idx, max_len, beam_size, memory_key_padding_mask)

    def _greedy(self, memory, bos_idx, eos_idx, max_len, mem_mask=None):
        B      = memory.size(0)
        tokens = torch.full((B, 1), bos_idx, dtype=torch.long, device=memory.device)
        for _ in range(max_len):
            tgt  = self.pos_enc(self.embedding(tokens) * math.sqrt(self.embed_dim))
            T    = tokens.size(1)
            mask = self._make_causal_mask(T, memory.device)
            out  = self.decoder(tgt, memory, tgt_mask=mask,
                                memory_key_padding_mask=mem_mask)
            nxt  = self.out_proj(out[:, -1]).argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
            if (nxt == eos_idx).all():
                break
        return tokens[:, 1:]  # strip <bos>

    def _beam(self, memory, bos_idx, eos_idx, max_len, beam_size, mem_mask=None):
        return self._greedy(memory, bos_idx, eos_idx, max_len, mem_mask)

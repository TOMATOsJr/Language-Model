"""
Architecture for the actual Decoder-only Transformer Model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            dmodel: int = 512,
            num_heads: int = 8,
            max_seq_len: int = 4096,
        ):
        super().__init__()
        self.dmodel = dmodel  # hidden dimension
        self.num_heads = num_heads  # number of attention heads
        self.dhead = self.dmodel // self.num_heads  # dimension of each attention head
        # The splitting of the q, k, v vectors into multiple
        # heads will be done in the forward pass.
        self.Wq = nn.Linear(self.dmodel, self.dmodel)
        self.Wk = nn.Linear(self.dmodel, self.dmodel)
        self.Wv = nn.Linear(self.dmodel, self.dmodel)
        self.Wo = nn.Linear(self.dmodel, self.dmodel)

    def forward(self, x, mask=None):
        B, T, _ = x.size() # here _ is the hidden dimension (dmodel)
        # Linear projections and splitting.
        q = self.Wq(x).view(B, T, self.num_heads, self.dhead).transpose(1, 2)  # (B, num_heads, T, dhead)
        k = self.Wk(x).view(B, T, self.num_heads, self.dhead).transpose(1, 2)  # (B, num_heads, T, dhead)
        v = self.Wv(x).view(B, T, self.num_heads, self.dhead).transpose(1, 2)  # (B, num_heads, T, dhead)

        # Scaled dot-product attention
        attn_scores = q @ k.transpose(-2, -1) / (math.sqrt(self.dhead))  # (B, num_heads, T, T)
        if mask is not None: # Pass the causal mask tensor for the attention mechanism
            # mask = torch.tril(torch.ones(T, T, device=q.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            # Apply causal mask (the -inf) and get the logits.
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf')) # (B, num_heads, T, T)
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, num_heads, T, T)
        attn_output = attn_probs @ v  # (B, num_heads, T, dhead)
        # Concatenate and apply the final linear projection.
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.dmodel)  # (B, T, dmodel)
        return self.Wo(attn_output)  # (B, T, dmodel)

class FeedForward(nn.Module):
    def __init__(
            self,
            dmodel: int,
            dff: int,
            dropout: float,
        ):
        super().__init__()
        self.dmodel = dmodel
        self.dff = dff  # dimension of the feed-forward layer
        self.W1 = nn.Linear(self.dmodel, self.dff)
        self.non_linearity = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.W2 = nn.Linear(self.dff, self.dmodel)

    def forward(self, x):
        return self.W2(self.dropout(self.non_linearity(self.W1(x)))) # (B, T, dmodel)

class DecoderLayer(nn.Module):
    """
    A single decoder layer consisting of a multi-head self-attention block
    followed by a feed-forward block, each with residual connections and
    pre-layer normalization.
    """
    def __init__(
            self,
            dmodel: int,
            num_heads: int,
            dff: int,
            dropout: float,
        ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(dmodel=dmodel, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForward(dmodel=dmodel, dff=dff, dropout=dropout)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)

    def forward(self, x, mask=None): # mask is the causal mask for the attention mechanism
        # Pre-layer normalization
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=mask))  # (B, T, dmodel)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))  # (B, T, dmodel)
        return x  # (B, T, dmodel)

class TransfomerDecoder(nn.Module):
    """
    A transformer decoder consisting of multiple decoder layers.
    """
    def __init__(
            self,
            num_layers: int,
            dmodel: int,
            num_heads: int,
            dff: int,
            dropout: float,
        ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(dmodel=dmodel, num_heads=num_heads, dff=dff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x  # (B, T, dmodel)

class TransformerLM(nn.Module):
    """Full decoder-only LM: token embedding -> decoder stack -> tied output projection."""
    def __init__(
            self,
            vocab_size: int,
            dmodel: int,
            num_layers: int,
            num_heads: int,
            dff: int,
            dropout: float,
            # max_seq_len: int,
            pad_id: int = 0,
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dmodel = dmodel
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, dmodel, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.decoder = TransfomerDecoder(
            num_layers=num_layers,
            dmodel=dmodel,
            num_heads=num_heads,
            dff=dff,
            dropout=dropout,
            # max_seq_len=max_seq_len,
        )
        self.final_norm = nn.LayerNorm(dmodel)

        # Output projection reuses the embedding matrix; no separate lm_head weight exists.
        self.lm_head = nn.Linear(dmodel, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.size()

        # Causal mask: position t may attend to positions <= t only.
        causal_mask = torch.tril(
            torch.ones(T, T, device=input_ids.device, dtype=torch.bool)
        ).view(1, 1, T, T)

        x = self.dropout(self.token_embedding(input_ids))  # (B, T, dmodel)
        x = self.decoder(x, mask=causal_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Tuple, List

class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention block
        att = self.norm1(src)
        att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        att = src + self.dropout1(att)

        # Feedforward block
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout2(out)
        return out


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Self attention block
        query = self.norm1(tgt)
        query = self.self_attn(
            query,
            query,
            query,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        query = tgt + self.dropout1(query)

        # Context attention block
        att = self.norm2(query)
        att = self.multihead_attn(
            att,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        att = query + self.dropout2(att)

        # Feedforward block
        out = self.norm3(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout3(out)
        return out

class CacheEnabledPreNormDecoderLayer(nn.Module):
    """
    A Transformer Decoder layer that uses 'pre-norm' normalization and
    supports a self-attention KV cache for efficient inference.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # Feed-forward implementation
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation function
        self.activation = getattr(nn.functional, activation)

    def forward(
       self, tgt: torch.Tensor, memory: torch.Tensor,
       tgt_mask: Optional[torch.Tensor] = None,
       memory_mask: Optional[torch.Tensor] = None,
       tgt_key_padding_mask: Optional[torch.Tensor] = None,
       memory_key_padding_mask: Optional[torch.Tensor] = None,
       past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
   ):
        q_k_v = self.norm1(tgt)

        # If a cache exists, tgt is only the last token.
        # Otherwise, it's the full sequence.
        q = k = v = q_k_v

        # KV CACHE LOGIC
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=0)
            v = torch.cat([past_v, v], dim=0)

        new_kv = (k, v)

        sa_output = self.self_attn(q, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(sa_output)


        mha_input = self.norm2(tgt)
        mha_output = self.multihead_attn(
            mha_input, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(mha_output)


        ffn_input = self.norm3(tgt)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(ffn_input))))
        tgt = tgt + self.dropout3(ffn_output)

        return tgt, new_kv

class CacheEnabledDecoder(nn.Module):
    """A stack of CacheEnabledDecoderLayers."""
    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        output = tgt
        new_kv_cache = []

        for i, layer in enumerate(self.layers):
            past_kv_for_layer = past_kv_cache[i] if past_kv_cache else None
            output, new_kv = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                past_kv=past_kv_for_layer
            )
            new_kv_cache.append(new_kv)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output, new_kv_cache

# vim: ts=4 sw=4 expandtab

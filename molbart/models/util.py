import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch.optim.lr_scheduler import LambdaLR

from typing import Callable, Union, Optional, Dict, Tuple, List

import copy

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

class RBFExpansion(nn.Module):
    """
    Expands scalar distances into a vector representation using Gaussian radial basis functions.
    This is a key component of SchNet and DimeNet.
    
    Args:
        low (float): Smallest distance to embed.
        high (float): Largest distance to embed.
        num_basis (int): The number of basis functions (output dimension).
        trainable (bool): Whether the RBF centers and gammas are trainable.
    """
    def __init__(self, low: float = 0.0, high: float = 30.0, num_basis: int = 64, trainable: bool = False):
        super().__init__()
        self.num_basis = num_basis
        
        # Initialize offsets (centers) and gammas
        offsets = torch.linspace(low, high, num_basis)
        spacing = offsets[1] - offsets[0]

        # gamma = 1 / (2 * sigma^2) 
        # sigma = spacing
        gammas = torch.tensor([0.5 / (spacing ** 2)] * num_basis)
        
        if trainable:
            self.offsets = nn.Parameter(offsets)
            self.gammas = nn.Parameter(gammas)
        else:
            self.register_buffer("offsets", offsets)
            self.register_buffer("gammas", gammas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)

        diff = dist - self.offsets.view(1, -1)

        # Gaussian RBF formula: exp(-gamma * (dist - offset)^2)
        return torch.exp(-self.gammas.view(1, -1) * (diff ** 2))

class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
    y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        glu_variant: bool = False,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: number of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.glu_variant = glu_variant
        if glu_variant:
            out_features = out_features * 2
        super().__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        if self.glu_variant:
            gate, y = F.linear(input, self.weight, self.bias).chunk(2, dim=-1)
            y = self.activation(gate) * y
        else:
            y = F.linear(input, self.weight, self.bias)
            y = self.activation(y)

        return y

def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Sum over values with the same indices.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        reduced input

    """
    return _scatter_add(x, idx_i, dim_size, dim)


@torch.jit.script
def _scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y

def scatter_mean(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    """
    Average over values with the same indices.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        reduced input (mean)

    """
    return _scatter_mean(x, idx_i, dim_size, dim)


@torch.jit.script
def _scatter_mean(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor:
    sums = _scatter_add(x, idx_i, dim_size, dim)

    count_shape = list(x.shape)
    for i in range(len(count_shape)):
        if i != dim:
            count_shape[i] = 1
    
    ones = torch.ones(count_shape, dtype=x.dtype, device=x.device)
    counts = _scatter_add(ones, idx_i, dim_size, dim)

    # Handle division by zero for indices that have no elements
    counts = counts.clamp(min=1)

    return sums / counts


# vim: ts=4 sw=4 expandtab

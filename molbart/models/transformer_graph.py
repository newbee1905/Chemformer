import math
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from molbart.models import BARTModel
from molbart.models.graph_models import PaiNNInteraction, PaiNNMixing, PaiNNGraphReinforcer, CosineCutoff, GaussianRBF
from molbart.models.util import PreNormDecoderLayer, PreNormEncoderLayer

from typing import Optional, Tuple, Dict, List, Callable

class GraphReinforcedEncoderLayer(nn.Module):
    """
    A Pre-Normalization Transformer Encoder Layer that injects
    PaiNN-based graph reinforcement before the multi-head attention block.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        dropout: float,
        activation: Callable,
        # PaiNN parameters
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        painn_activation: Callable,
        glu_variant: bool,
        epsilon: float,
    ):
        super().__init__()
        
        # Standard Transformer components
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=False)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Graph Reinforcement component for this layer
        self.painn_reinforcer = PaiNNGraphReinforcer(
            d_model=d_model,
            n_interactions=n_interactions,
            radial_basis=radial_basis,
            cutoff_fn=cutoff_fn,
            painn_activation=painn_activation,
            glu_variant=glu_variant,
            epsilon=epsilon,
        )

    def _run_gnn_reinforcement(
        self,
        src: torch.Tensor,
        gnn_data_list: List[Optional[Dict[str, torch.Tensor]]],
        atom_mask_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies PaiNN reinforcement to the atom tokens within the src tensor.
        
        Args:
            src (Tensor): (seq_len, batch_size, d_model)
            gnn_data_list (list): List of GNN data dicts
            atom_mask_tokens (Tensor): (seq_len, batch_size)
        
        Returns:
            reinforced_src (Tensor): (seq_len, batch_size, d_model)
        """
        
        reinforced_src = src.clone()
        seq_len, batch_size, d_model = src.shape
        
        for b_idx in range(batch_size):
            gnn_data = gnn_data_list[b_idx]

            # Get atom token mask for this sequence
            atom_mask = atom_mask_tokens[:, b_idx] # (seq_len,)
            atom_positions = torch.where(atom_mask == 1)[0]

            # Extract token embeddings for heavy atoms
            heavy_atom_token_embs = src[atom_positions, b_idx, :]

            # Run PaiNN to get graph-reinforced representations
            graph_reinforced_features = self.painn_reinforcer(
                    gnn_data, heavy_atom_token_embs
                    ) # (n_heavy_atoms, d_model)

            reinforced_src[atom_positions, b_idx, :] = graph_reinforced_features

        return reinforced_src

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,

        # GNN-specific data
        gnn_data_list: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        atom_mask_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        x = src # (seq_len, batch_size, d_model)
        x_norm1 = self.norm1(x)
        
        x_reinforced = self._run_gnn_reinforcement(
            x_norm1, gnn_data_list, atom_mask_tokens
        )
        
        attn_output, _ = self.self_attn(
            x_reinforced, 
            x_reinforced, 
            x_reinforced,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        
        x = x + self.dropout1(attn_output)
        x_norm2 = self.norm2(x)

        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(x_norm2))))
        x = x + self.dropout2(ffn_output)
        
        return x

class GraphReinforcedEncoder(nn.TransformerEncoder):
    def __init__(self, layers_list, num_layers, norm=None):
        nn.Module.__init__(self) 
        self.layers = layers_list
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        gnn_data_list: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        atom_mask_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src

        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                gnn_data_list=gnn_data_list,
                atom_mask_tokens=atom_mask_tokens,
            )
        if self.norm is not None:
            output = self.norm(output)

        return output


class GraphBARTModel(BARTModel):
    """
    BART model where the encoder is a stack of GraphReinforcedEncoderLayers.
    
    Each layer in the encoder applies a small PaiNN GNN to the atom
    representations before the multi-head attention step.
    """
    
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocabulary_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,

        # PaiNN specific parameters
        n_interactions=2,  # Using a "small PaiNN" per layer
        radial_basis=GaussianRBF(n_rbf=20, cutoff=5.0),
        cutoff_fn=CosineCutoff(5.0),
        painn_activation=F.silu,
        glu_variant=True,
        epsilon=1e-8,
        # Other parameters
        schedule="cycle",
        warm_up_steps=None,
        optimizer="adam",
        dropout=0.3,
        **kwargs,
    ):
        # Initialize parent BART model
        super().__init__(
            decode_sampler=decode_sampler,
            pad_token_idx=pad_token_idx,
            vocabulary_size=vocabulary_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            lr=lr,
            weight_decay=weight_decay,
            activation=activation,
            num_steps=num_steps,
            max_seq_len=max_seq_len,
            schedule=schedule,
            warm_up_steps=warm_up_steps,
            optimizer=optimizer,
            dropout=dropout,
            **kwargs,
        )
        
        # Store GNN components
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn

        # Create the stack of layers, each with its own weights
        encoder_layers_list = nn.ModuleList([
            GraphReinforcedEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_feedforward=d_feedforward,
                dropout=dropout,
                activation=self.activation,
                n_interactions=n_interactions,
                radial_basis=self.radial_basis,
                cutoff_fn=self.cutoff_fn,
                painn_activation=painn_activation,
                glu_variant=glu_variant,
                epsilon=epsilon,
            ) for _ in range(num_layers)
        ])
        
        self.encoder = GraphReinforcedEncoder(
            layers_list=encoder_layers_list,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.decoder = nn.TransformerDecoder(
            PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        self._init_params()

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Enhanced encoding with graph reinforcement *inside* the encoder stack.
        """
        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        
        embs = self._construct_input(encoder_input)
        
        memory = self.encoder(
            embs,
            src_key_padding_mask=encoder_pad_mask,
            # Pass GNN data
            gnn_data_list=batch.get("gnn_data_list"),
            atom_mask_tokens=batch.get("atom_mask_tokens"),
        )
        
        return memory

# vim: ts=4 sw=4 expandtab

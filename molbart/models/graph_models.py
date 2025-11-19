import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy

from typing import Callable, Optional, List, Dict

from molbart.models.util import Dense, scatter_add


class PaiNNInteraction(nn.Module):
    r"""PaiNN interaction block for modeling equivariant interactions of atomistic systems."""

    def __init__(self, n_atom_basis: int, activation: Callable, glu_variant: bool):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
        """
        super(PaiNNInteraction, self).__init__()

        self.n_atom_basis = n_atom_basis

        self.interatomic_context_net = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=activation, glu_variant=glu_variant),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        Wij: torch.Tensor,
        dir_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int,
    ):
        """Compute interaction output.

        Args:
            q: scalar input values
            mu: vector input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        # inter-atomic
        x = self.interatomic_context_net(q)
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = scatter_add(dq, idx_i, dim_size=n_atoms)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = scatter_add(dmu, idx_i, dim_size=n_atoms)

        q = q + dq
        mu = mu + dmu

        return q, mu


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8, glu_variant: bool = False):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis

        self.intraatomic_context_net = nn.Sequential(
            Dense(2 * n_atom_basis, n_atom_basis, activation=activation, glu_variant=glu_variant),
            Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_channel_mix = Dense(
            n_atom_basis, 2 * n_atom_basis, activation=None, bias=False
        )
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """Compute intraatomic mixing.

        Args:
            q: scalar input values
            mu: vector input values

        Returns:
            atom features after interaction
        """
        ## intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu

class PaiNNGraphReinforcer(nn.Module):
    """
    A self-contained module that applies PaiNN reinforcement to a set
    of heavy-atom embeddings, given corresponding 3D graph data.
    """
    def __init__(
        self,
        d_model: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        painn_activation: Callable = F.silu,
        glu_variant: bool = True,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_interactions = n_interactions
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        
        # Hydrogen embedding - stays within PaiNN
        self.hydrogen_embedding = nn.Embedding(1, d_model)
        
        # Filter network
        if shared_filters:
            self.filter_net = Dense(
                self.radial_basis.n_rbf, 3 * d_model, activation=None
            )
        else:
            self.filter_net = Dense(
                self.radial_basis.n_rbf,
                self.n_interactions * d_model * 3,
                activation=None,
            )
        
        self.painn_interactions = nn.ModuleList([
            PaiNNInteraction(
                n_atom_basis=d_model,
                activation=painn_activation,
                glu_variant=glu_variant
            )
            for _ in range(n_interactions)
        ])
        
        self.painn_mixing = nn.ModuleList([
            PaiNNMixing(
                n_atom_basis=d_model,
                activation=painn_activation,
                epsilon=epsilon,
                glu_variant=glu_variant
            )
            for _ in range(n_interactions)
        ])

    def forward(
        self, 
        gnn_data: Dict[str, torch.Tensor],
        heavy_atom_token_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Process molecular graph through PaiNN using token embeddings as input.
        
        Args:
            gnn_data: Dictionary containing graph structure information
            heavy_atom_token_embeddings: Token embeddings for heavy atoms (n_heavy_atoms, d_model)
        
        Returns:
            q: reinforced heavy atom representations (n_heavy_atoms, d_model)
        """

        atomic_numbers = gnn_data['Z']
        positions = gnn_data['R']
        idx_i = gnn_data['idx_i']
        idx_j = gnn_data['idx_j']
        gnn_is_heavy = gnn_data['gnn_is_heavy_atom_mask']
        n_atoms = atomic_numbers.shape[0]
        
        # Compute pairwise features
        r_ij = positions[idx_j] - positions[idx_i]
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / (d_ij + 1e-8)
        
        # Radial basis and cutoff
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)
        
        # Compute filters
        filters = self.filter_net(phi_ij) * fcut[..., None]
        filter_list = torch.split(filters, 3 * self.d_model, dim=-1)
        
        # Initialize scalar features 'q'
        q = torch.zeros(
            (n_atoms, self.d_model),
            device=heavy_atom_token_embeddings.device, 
            dtype=heavy_atom_token_embeddings.dtype,
        )
        
        # Place heavy atom embeddings
        heavy_indices = torch.where(gnn_is_heavy.bool())[0]
        q[heavy_indices] = heavy_atom_token_embeddings
        
        # Place hydrogen embeddings
        is_hydrogen = (atomic_numbers == 1)
        hydrogen_indices = torch.where(is_hydrogen)[0]
        if len(hydrogen_indices) > 0:
            h_emb = self.hydrogen_embedding(
                torch.zeros(
                    len(hydrogen_indices),
                    dtype=torch.long, 
                    device=q.device,
                ),
            )
            q[hydrogen_indices] = h_emb
        
        # Initialize vector features
        mu = torch.zeros((n_atoms, 3, self.d_model), device=q.device)
        
        # PaiNN message passing
        for i, (interaction, mixing) in enumerate(
            zip(self.painn_interactions, self.painn_mixing)
        ):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)
        
        q_heavy_out = q[heavy_indices]
        return q_heavy_out

def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor):
    r""" Behler-style cosine cutoff.

        .. math::
           f(r) = \begin{cases}
            0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
              & r < r_\text{cutoff} \\
            0 & r \geqslant r_\text{cutoff} \\
            \end{cases}

        Args:
            cutoff (float, optional): cutoff radius.

        """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).float()
    return input_cut


class CosineCutoff(nn.Module):
    r""" Behler-style cosine cutoff module.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float, optional): cutoff radius.
        """
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)

def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)

# vim:ts=4 sw=4 et

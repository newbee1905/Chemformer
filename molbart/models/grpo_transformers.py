from __future__ import annotations

import torch
from copy import deepcopy
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch.optim.lr_scheduler import OneCycleLR
from collections import Counter

from molbart.models.transformer_models import BARTModel
from typing import List, Dict

# Suppress RDKit console output
RDLogger.DisableLog("rdApp.*")

class BARTGRPOModel(BARTModel):
    """
    A BARTModel subclass that overrides the training process for Reinforcement Learning
    using a GRPO algorithm.
    """

    automatic_optimization = False

    def __init__(self, rl_learning_rate: float, ppo_epochs: int, clip_param: float, kl_coeff: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("rl_learning_rate", "ppo_epochs", "clip_param", "kl_coeff", "g_samples")

        # self.ref_model = deepcopy(self)
        full_state_dict = self.state_dict()
        policy_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith("ref_model.")}

        self.ref_model = None

        self.mfpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def setup(self, stage=None):
        if self.ref_model is None:
            print("Setting up and freezing the reference model...")
            
            policy_state_dict = {k: v for k, v in self.state_dict().items() if not k.startswith("ref_model.")}

            self.ref_model = BARTModel(decode_sampler=self.sampler, **self.hparams)
            self.ref_model.load_state_dict(policy_state_dict)
            
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def training_step(self, batch: Dict, batch_idx: int):
        opt = self.optimizers()
        # Before 1.3, Lightning automatically called lr_scheduler.step() in both automatic and manual optimization.
        # From 1.3, lr_scheduler.step() is now for the user to call at arbitrary intervals.
        sch = self.lr_schedulers()

        original_target_smiles = batch["target_smiles"]
        original_encoder_input = batch["encoder_input"]
        original_encoder_pad_mask = batch["encoder_pad_mask"]

        self.sampler.gumbel_noise = True
        self.eval()

        all_sampled_smiles = []
        all_log_probs_old = []
        all_target_smiles_repeated = []
        original_indices = []

        for _ in range(self.hparams.g_samples):
            sampled_smiles_k, log_probs_old_k = self.sample_molecules(batch, sampling_alg="beam")

            valid_idx_this_round = [i for i, s in enumerate(sampled_smiles_k) if len(s) > 0]
            if not valid_idx_this_round:
                continue

            all_sampled_smiles.extend([sampled_smiles_k[i][0] for i in valid_idx_this_round])
            all_log_probs_old.extend([log_probs_old_k[i][0] for i in valid_idx_this_round])
            
            all_target_smiles_repeated.extend([original_target_smiles[i] for i in valid_idx_this_round])
            
            original_indices.extend(valid_idx_this_round)
        
        if not all_sampled_smiles:
            return

        log_probs_old = torch.tensor(all_log_probs_old, device=self.device)
        rewards = self._reward_fn(all_sampled_smiles, all_target_smiles_repeated)

        # Remove std normalisation from GRPO based on Dr. GRPO
        # https://arxiv.org/abs/2503.20783
        advantage = rewards - rewards.mean()

        encoded_tokens = [t.tolist() for t in self.sampler.tokenizer.encode(all_sampled_smiles, enclose=False)]
        pad_token_id = self.sampler.tokenizer[self.sampler.tokenizer.special_tokens["pad"]]
        max_len = self.hparams.max_seq_len

        padded_ids = []
        attention_masks = []
        for tokens in encoded_tokens:
            truncated_tokens = tokens[:max_len]
            padding_needed = max_len - len(truncated_tokens)
            
            padded_ids.append(truncated_tokens + [pad_token_id] * padding_needed)
            attention_masks.append([1] * len(truncated_tokens) + [0] * padding_needed)
        
        target_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)[:, 1:].transpose(0, 1)
        decoder_input = torch.tensor(padded_ids, dtype=torch.long, device=self.device)[:, :-1].transpose(0, 1)

        target_mask = torch.tensor(attention_masks, dtype=torch.bool, device=self.device)[:, 1:].transpose(0, 1)
        target_mask = ~target_mask

        decoder_mask = torch.tensor(attention_masks, dtype=torch.bool, device=self.device)[:, :-1].transpose(0, 1)
        decoder_mask = ~decoder_mask
        
        new_encoder_input = original_encoder_input[:, original_indices]
        new_encoder_pad_mask = original_encoder_pad_mask[:, original_indices]

        ppo_batch = {
            "encoder_input": new_encoder_input,
            "encoder_pad_mask": new_encoder_pad_mask,
            "decoder_input": decoder_input,
            "decoder_pad_mask": decoder_mask,
            "target": target_ids,
            "target_mask": target_mask,
        }

        self.train()
        
        for _ in range(self.hparams.ppo_epochs):
            log_probs_current = self._get_log_probs(self, ppo_batch)
            
            ratio = torch.exp(log_probs_current - log_probs_old)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.hparams.clip_param, 1.0 + self.hparams.clip_param) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            # --- Cumulative Sequence-Level KL Divergence Calculation ---
            # This implements the cumulative estimate from the paper for better credit assignment.
            # The gradient for a token at time 't' is scaled by the sum of log-ratios of all future tokens.
            # The full formula for the gradient estimate is:
            #
            # sum_{t=1 to T} [ (sum_{s=t to T} log(rho_s)) * grad(log(policy(y_t | y_{1:t-1}))) ]
            # Based on the formula from the paper: https://arxiv.org/abs/2506.09477

            per_token_log_probs_current = self._get_per_token_log_probs(self, ppo_batch)
            with torch.no_grad():
                per_token_log_probs_ref = self._get_per_token_log_probs(self.ref_model, ppo_batch)

            per_token_log_ratio = per_token_log_probs_current - per_token_log_probs_ref
            cumulative_log_ratio = torch.flip(torch.cumsum(torch.flip(per_token_log_ratio, dims=[0]), dim=0), dims=[0])

            kl_loss_terms = cumulative_log_ratio.detach() * per_token_log_probs_current
            kl_loss = kl_loss_terms.sum(dim=0).mean()

            loss = policy_loss + self.hparams.kl_coeff * kl_loss

            opt.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()

        # Don't need to call this for version before 1.3
        sch.step()

        self.sampler.gumbel_noise = False

        self.log_dict({
            "train_loss": loss.detach(),
            "mean_reward": rewards.mean(),
            "policy_loss": policy_loss.detach(),
            "kl_div": kl_loss.detach(),
        }, prog_bar=True)

    def configure_optimizers(self):
        """ Configure the optimizer for the policy model. """

        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if param.dim() == 1 or "bias" in name or "norm" in name.lower() or "ref" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': 0.1},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        optim = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.95),
        )

        sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)

        return [optim], [sch]

    def _get_atom_counts(self, smiles: str) -> Counter:
        """Calculates the count of each atom in a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return Counter()

        mol = Chem.AddHs(mol)
        return Counter(atom.GetSymbol() for atom in mol.GetAtoms())

    def _reward_fn(self, generated_smiles: List[str], target_smiles: List[str]) -> torch.Tensor:
        rewards = []
        for smi_gen, smi_target in zip(generated_smiles, target_smiles):
            mol_gen = Chem.MolFromSmiles(smi_gen)
            mol_target = Chem.MolFromSmiles(smi_target)
            if mol_gen is None or mol_target is None:
                rewards.append(0.0)
                continue

            counts_gen = self._get_atom_counts(smi_gen)
            counts_target = self._get_atom_counts(smi_target)
            if counts_gen != counts_target:
                rewards.append(0.0)
                continue
            
            fp_gen = self.mfpgen.GetFingerprint(mol_gen)
            fp_target = self.mfpgen.GetFingerprint(mol_target)

            rewards.append(DataStructs.TanimotoSimilarity(fp_gen, fp_target))
            
        return torch.tensor(rewards, device=self.device)

    def _get_per_token_log_probs(self, model: BARTModel, batch: Dict) -> torch.Tensor:
        """ Returns per-token log probabilities. Shape: (seq_len, batch_size) """
        target_ids = batch["target"]
        target_mask = batch["target_mask"]

        model_output = model(batch)

        log_probs_dist = model_output["token_output"]
        log_probs = log_probs_dist.gather(2, target_ids.unsqueeze(2)).squeeze(2)

        attention_mask = ~target_mask

        # Apply mask to zero-out padding tokens but don't sum over the sequence length
        return log_probs * attention_mask

    def _get_log_probs(self, model: BARTModel, batch: Dict) -> torch.Tensor:
        """ Returns sequence-level log probabilities. Shape: (batch_size,) """
        per_token_log_probs = self._get_per_token_log_probs(model, batch)

        return per_token_log_probs.sum(dim=0)

# vim: ts=4 sw=4 expandtab

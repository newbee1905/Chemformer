import math
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from molbart.models import _AbsTransformerModel, BARTModel
from molbart.models.util import PreNormDecoderLayer, PreNormEncoderLayer

class CycleConsistencyBARTModel(BARTModel):
    def __init__(self, *args, **kwargs):
        # self.w_retro = kwargs.pop("w_retro", 0.5)
        # self.w_cos = kwargs.pop("w_cos", 0.1)
        # self.gumbel_tau = kwargs.pop("gumbel_tau", 0.3)

        self.w_retro = 1.0
        self.w_cos = 0.0

        super().__init__(*args, **kwargs)

        self.cos_sim_fn = nn.CosineSimilarity(dim=-1)
        # self.loss_function = nn.CrossEntropyLoss(reduction="none", ignore_index=self.pad_token_idx, label_smoothing=0.1)

        self.max_tau = 1.2
        self.min_tau = 0.1
        self.max_consistency_weight = 0.1
        self.min_consistency_weight = 0.01

        self.tau_decay_steps = self.num_steps * 0.75
        self.consistency_warm_up = self.num_steps * 0.05
        self.consistency_decay_steps = (self.num_steps - self.consistency_warm_up) * 0.5

    def _get_eos_representation(
        self, 
        hidden_states: torch.Tensor, 
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        eos_token_id = self.sampler.tokenizer["end"]
        
        eos_indices = (target_ids == eos_token_id)
        if eos_indices.any():
            eos_locs = eos_indices.long().argmax(dim=0)
            return hidden_states[eos_locs, torch.arange(hidden_states.size(1))]
        else:
            # Fallback to last position
            return hidden_states[-1]

    def training_step(self, batch, batch_idx):
        """
        Overrides the original training_step to implement the composite loss.
        L_total = L_forward + w_retro * L_retro + w_cos * L_cos
        """

        forward_output = self.forward(batch)
        l_forward = self._calc_loss(batch, forward_output)

        # if self.global_step <= self.consistency_warm_up:
        #     current_consistency = 0.0
        # else:
        #     warmup_ratio = min((self.global_step - self.consistency_warm_up) / self.consistency_decay_steps, 1.0)
        #     current_consistency = self.min_consistency_weight * math.exp(math.log(self.max_consistency_weight / self.min_consistency_weight) * warmup_ratio)
        # self.log("w_consistency", current_consistency, on_step=True, logger=True)

        decay_ratio = min(self.global_step / self.tau_decay_steps, 1.0)
        current_tau = self.max_tau * math.exp(-math.log(self.max_tau / self.min_tau) * decay_ratio)
        self.log("gumbel_tau", current_tau, on_step=True, logger=True)
        # current_tau = self.gumbel_tau

        forward_logits = forward_output["token_output"]
        predicted_product_probs = F.gumbel_softmax(
            forward_logits, tau=current_tau, hard=False, dim=-1
        )

        soft_product_embs = torch.matmul(predicted_product_probs, self.emb.weight)
        soft_product_embs = soft_product_embs * math.sqrt(self.d_model)
        seq_len, _, _ = soft_product_embs.size()
        pos_embs = self.pos_emb[:seq_len, :].unsqueeze(1)
        retro_encoder_embs = self.dropout(soft_product_embs + pos_embs)

        retro_encoder_pad_mask = batch["target_mask"].clone().transpose(0, 1)
        retro_memory = self.encoder(retro_encoder_embs, src_key_padding_mask=retro_encoder_pad_mask)

        retro_decoder_input = batch["encoder_input"][:-1, :]
        retro_decoder_pad_mask = batch["encoder_pad_mask"][:-1, :].transpose(0, 1)
        retro_decoder_embs = self._construct_input(retro_decoder_input)

        tgt_seq_len, _, _ = retro_decoder_embs.size()
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device=self.device)

        retro_decoder_output = self.decoder(
            retro_decoder_embs,
            retro_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=retro_decoder_pad_mask,
            memory_key_padding_mask=retro_encoder_pad_mask.clone()
        )
        # retro_logits = self.token_fc(retro_decoder_output)
        # retro_output = {"model_output": retro_decoder_output, "token_output": retro_logits}

        # retro_loss_batch = {
        #     "target": batch["encoder_input"][1:, :],
        #     "target_mask": batch["encoder_pad_mask"][1:, :],
        # }
        # l_retro = self._calc_loss(retro_loss_batch, retro_output)

        # predicted_reactant_probs = F.gumbel_softmax(retro_logits, tau=current_tau, hard=False, dim=-1)
        # soft_reactant_embs = torch.matmul(predicted_reactant_probs, self.emb.weight)
        # soft_reactant_embs = soft_reactant_embs * math.sqrt(self.d_model)

        # seq_len, _, _ = soft_reactant_embs.size()
        # pos_embs_r2 = self.pos_emb[:seq_len, :].unsqueeze(1)
        # recon_encoder_embs = self.dropout(soft_reactant_embs + pos_embs_r2)
        
        # recon_encoder_pad_mask = batch["encoder_pad_mask"][1:, :].clone().transpose(0, 1)
        # recon_memory = self.encoder(recon_encoder_embs, src_key_padding_mask=recon_encoder_pad_mask)
        
        # recon_decoder_input = batch["decoder_input"]
        # recon_decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        # recon_decoder_embs = self._construct_input(recon_decoder_input)
        # tgt_seq_len, _, _ = recon_decoder_embs.size()
        # tgt_mask_r2 = self._generate_square_subsequent_mask(tgt_seq_len, device=self.device)
        
        # recon_decoder_output = self.decoder(
        #     recon_decoder_embs,
        #     recon_memory,
        #     tgt_mask=tgt_mask_r2,
        #     tgt_key_padding_mask=recon_decoder_pad_mask,
        #     memory_key_padding_mask=recon_encoder_pad_mask.clone()
        # )
        # p2_logits = self.token_fc(recon_decoder_output)
        # l_p2 = self._calc_loss(batch, {
        #     "model_output": recon_decoder_output,
        #     "token_output": p2_logits,
        # })

        # l_consistency = F.kl_div(
        #     F.log_softmax(p2_logits, dim=-1),
        #     F.softmax(forward_logits, dim=-1),
        #     reduction='batchmean'
        # )

        # l_total = l_forward + l_p2 + (self.w_retro * l_retro) + (current_consistency * l_consistency)

        # self.log_dict({
        #     "train_loss_total": l_total,
        #     "train_loss_forward": l_forward,
        #     "train_loss_retro": l_retro,
        #     "train_loss_forward_2": l_p2,
        #     "train_loss_consistency": l_consistency,
        # }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # return l_total

        retro_token_output = self.token_fc(retro_decoder_output)
        retro_output = {"model_output": retro_decoder_output, "token_output": retro_token_output}

        retro_loss_batch = {
            "target": batch["encoder_input"][1:, :],
            "target_mask": batch["encoder_pad_mask"][1:, :],
        }
        l_retro = self._calc_loss(retro_loss_batch, retro_output)

        forward_eos_repr = self._get_eos_representation(forward_output["model_output"], batch["target"])
        retro_eos_repr = self._get_eos_representation(retro_output["model_output"], retro_loss_batch["target"])
        l_cos = 1.0 - self.cos_sim_fn(forward_eos_repr, retro_eos_repr.detach()).mean()

        l_total = l_forward + (self.w_retro * l_retro) + (self.w_cos * l_cos)

        self.log_dict({
            "train_loss_total": l_total,
            "train_loss_forward": l_forward,
            "train_loss_retro": l_retro,
            "train_loss_cos": l_cos
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return l_total


class CycleConsistencySepBARTModel(CycleConsistencyBARTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_retro_input(self, soft_product_embs, batch):
        device = soft_product_embs.device
        product_lengths = batch["product_lengths"]
        retro_ids = batch["retro_ids"]

        seq_len_prod = soft_product_embs.size(0)
        seq_len_retro = retro_ids.size(0)

        retro_embs_template = self.emb(retro_ids) * math.sqrt(self.d_model)

        src_prod_mask = torch.arange(seq_len_prod, device=device)[:, None] < product_lengths[None, :]
        dest_prod_mask = torch.arange(seq_len_retro, device=device)[:, None] < product_lengths[None, :]

        final_retro_embs = retro_embs_template
        final_retro_embs[dest_prod_mask] = soft_product_embs[src_prod_mask]
        
        pos_embs = self.pos_emb[:seq_len_retro, :].unsqueeze(1)
        final_retro_embs = self.dropout(final_retro_embs + pos_embs)

        return final_retro_embs

    def training_step(self, batch, batch_idx):
        """
        Overrides the original training_step to implement the composite loss.
        L_total = L_forward + w_retro * L_retro + w_cos * L_cos
        """

        forward_output = self.forward(batch)
        l_forward = self._calc_loss(batch, forward_output)

        # decay_ratio = min(self.global_step / self.tau_decay_steps, 1.0)
        # current_tau = self.max_tau * math.exp(-math.log(self.max_tau / self.min_tau) * decay_ratio)
        # self.log("gumbel_tau", current_tau, on_step=True, logger=True)
        current_tau = self.gumbel_tau

        forward_logits = forward_output["token_output"]
        predicted_product_probs = F.gumbel_softmax(
            forward_logits, tau=current_tau, hard=False, dim=-1
        )

        soft_product_embs = torch.matmul(predicted_product_probs, self.emb.weight)
        soft_product_embs = soft_product_embs * math.sqrt(self.d_model)
        seq_len, _, _ = soft_product_embs.size()
        pos_embs = self.pos_emb[:seq_len, :].unsqueeze(1)
        soft_product_embs = self.dropout(soft_product_embs + pos_embs)

        retro_encoder_embs = self._prepare_retro_input(soft_product_embs, batch)
        retro_encoder_pad_mask = batch["retro_mask"].transpose(0, 1)
        retro_memory = self.encoder(retro_encoder_embs, src_key_padding_mask=retro_encoder_pad_mask)

        retro_decoder_input = batch["reactants_ids"][:-1, :]
        retro_decoder_pad_mask = batch["reactants_mask"][:-1, :].transpose(0, 1)
        retro_decoder_embs = self._construct_input(retro_decoder_input)

        tgt_seq_len, _, _ = retro_decoder_embs.size()
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device=self.device)

        retro_decoder_output = self.decoder(
            retro_decoder_embs,
            retro_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=retro_decoder_pad_mask,
            memory_key_padding_mask=retro_encoder_pad_mask,
        )
        retro_token_output = self.token_fc(retro_decoder_output)
        retro_output = {"model_output": retro_decoder_output, "token_output": retro_token_output}

        retro_loss_batch = {
            "target": batch["reactants_ids"][1:, :],
            "target_mask": batch["reactants_mask"][1:, :],
        }
        l_retro = self._calc_loss(retro_loss_batch, retro_output)

        forward_eos_repr = self._get_eos_representation(forward_output["model_output"], batch["target"])
        retro_eos_repr = self._get_eos_representation(retro_output["model_output"], retro_loss_batch["target"])
        l_cos = 1.0 - self.cos_sim_fn(forward_eos_repr, retro_eos_repr.detach()).mean()

        l_total = l_forward + (self.w_retro * l_retro) + (self.w_cos * l_cos)

        self.log_dict({
            "train_loss_total": l_total,
            "train_loss_forward": l_forward,
            "train_loss_retro": l_retro,
            "train_loss_cos": l_cos
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return l_total

# vim: ts=4 sw=4 expandtab

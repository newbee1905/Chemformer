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
        self.w_retro = kwargs.pop("w_retro", 1.0)
        self.w_cos = kwargs.pop("w_cos", 0.5)
        # self.gumbel_tau = kwargs.pop("gumbel_tau", 1.0)

        super().__init__(*args, **kwargs)

        self.cos_sim_fn = nn.CosineSimilarity(dim=-1)
        self.loss_function = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx, label_smoothing=0.1)

        self.max_tau = 2.0
        self.min_tau = 0.5

        self.tau_decay_steps = self.num_steps * 0.75

    def _get_eos_representation(
        self, 
        hidden_states: torch.Tensor, 
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback EOS representation extraction with better handling."""

        eos_token_id = getattr(
            self.sampler.tokenizer, 'eos_token_id', 
            self.sampler.tokenizer.token_to_idx.get("end", -1)
        )
        
        # No EOS token defined, use last non-padded position
        if eos_token_id == -1:
            mask = target_ids != self.pad_token_idx
            lengths = mask.sum(dim=0) - 1
            lengths = torch.clamp(lengths, min=0, max=hidden_states.size(0)-1)
            return hidden_states[lengths, torch.arange(hidden_states.size(1))]
        
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

        decay_ratio = min(self.global_step / self.tau_decay_steps, 1.0)
        current_tau = self.max_tau * math.exp(-math.log(self.max_tau / self.min_tau) * decay_ratio)
        self.log("gumbel_tau", current_tau, on_step=True, logger=True)

        forward_logits = forward_output["token_output"]
        predicted_product_probs = F.gumbel_softmax(
            forward_logits, tau=current_tau, hard=False, dim=-1
        )

        soft_product_embs = torch.matmul(predicted_product_probs, self.emb.weight)
        soft_product_embs = soft_product_embs * math.sqrt(self.d_model)
        seq_len, _, _ = soft_product_embs.size()
        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(1)
        retro_encoder_embs = self.dropout(soft_product_embs + positional_embs)

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

    def training_step(self, batch, batch_idx):
        """
        Overrides the original training_step to implement the composite loss.
        L_total = L_forward + w_retro * L_retro + w_cos * L_cos
        """

        forward_output = self.forward(batch)
        l_forward = self._calc_loss(batch, forward_output)

        decay_ratio = min(self.global_step / self.tau_decay_steps, 1.0)
        current_tau = self.max_tau * math.exp(-math.log(self.max_tau / self.min_tau) * decay_ratio)
        self.log("gumbel_tau", current_tau, on_step=True, logger=True)

        forward_logits = forward_output["token_output"]
        predicted_product_probs = F.gumbel_softmax(
            forward_logits, tau=current_tau, hard=False, dim=-1
        )

        soft_product_embs = torch.matmul(predicted_product_probs, self.emb.weight)
        soft_product_embs = soft_product_embs * math.sqrt(self.d_model)
        seq_len, _, _ = soft_product_embs.size()
        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(1)
        retro_encoder_embs = self.dropout(soft_product_embs + positional_embs)

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

class DifferentiableCycleBARTModel(BARTModel):
    """
    A BART-style model that uses a fully differentiable cycle-consistency loss
    for training. 

    It avoids the non-differentiable `argmax` operation by creating
    "soft" embeddings of the predicted product, allowing gradients to flow
    end-to-end.
    """
    def __init__(self, *args, **kwargs):
        self.w_cycle = kwargs.pop("w_cycle", 1.0)
        self.w_kl = kwargs.pop("w_kl", 0.1)
        self.temperature = kwargs.pop("temperature", 1.0)
        
        super().__init__(*args, **kwargs)

    def _logits_to_soft_embeddings(self, logits):
        """
        Converts decoder logits into a differentiable, "soft" embedding sequence.
        """

        probs = F.softmax(logits / self.temperature, dim=-1)
        token_embeddings = self.emb.weight
        
        # Compute soft embeddings via matrix multiplication (weighted average)
        # (seq, batch, vocab_size) @ (vocab_size, d_model) -> (seq, batch, d_model)
        soft_embeddings = torch.matmul(probs, token_embeddings)
        
        return soft_embeddings

    def _decode_from_soft_embeddings(
        self, soft_encoder_embeddings, decoder_input,
        decoder_pad_mask, encoder_pad_mask,
    ):
        """
        Performs a decoder pass using soft embeddings directly as the encoder memory.
        This is used for the reverse step of the cycle.
        """

        decoder_embeddings = self._construct_input(decoder_input)
        decoder_pad_mask = decoder_pad_mask.transpose(0, 1)
        encoder_pad_mask = encoder_pad_mask.transpose(0, 1)
        
        seq_len, _, _ = decoder_embeddings.size()
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_embeddings.device)
        
        decoder_output = self.decoder(
            decoder_embeddings,
            soft_encoder_embeddings,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=encoder_pad_mask,
            tgt_mask=tgt_mask,
        )
        
        token_output = self.token_fc(decoder_output)
        return {
            "model_output": decoder_output,
            "token_output": token_output,
        }

    def training_step(self, batch, batch_idx):
        """
        Implements the fully differentiable composite loss.
        L_total = L_forward + w_cycle * L_cycle + w_kl * L_KL
        """

        forward_output = self.forward(batch)
        l_forward = self._calc_loss(batch, forward_output)

        forward_logits = forward_output["token_output"]
        soft_product_embeddings = self._logits_to_soft_embeddings(forward_logits)

        reverse_output = self._decode_from_soft_embeddings(
            soft_encoder_embeddings=soft_product_embeddings,
            decoder_input=batch["encoder_input"],
            decoder_pad_mask=batch["encoder_pad_mask"],
            encoder_pad_mask=batch["decoder_pad_mask"]
        )

        cycle_target_batch = {
            "target": batch["encoder_input"],
            "target_mask": batch["encoder_pad_mask"]
        }
        l_cycle = self._calc_loss(cycle_target_batch, reverse_output)

        forward_decoder_memory = forward_output["model_output"]
        batch_size = forward_decoder_memory.size(1)
        forward_lengths = (~batch["decoder_pad_mask"]).sum(dim=0)
        forward_last_indices = forward_lengths - 1
        
        product_last_token_repr = forward_decoder_memory[forward_last_indices, torch.arange(batch_size)]

        reverse_decoder_memory = reverse_output["model_output"]
        reverse_lengths = (~batch["encoder_pad_mask"]).sum(dim=0)
        reverse_last_indices = reverse_lengths - 1
        
        reactant_last_token_repr = reverse_decoder_memory[reverse_last_indices, torch.arange(batch_size)]

        p_dist = F.softmax(product_last_token_repr.detach(), dim=-1)
        q_dist = F.softmax(reactant_last_token_repr, dim=-1)

        # Average distribution 'm'
        m_dist = 0.5 * (p_dist + q_dist)
        log_m_dist = m_dist.log()

        # KL(P || M)
        kl_p_m = F.kl_div(log_m_dist, p_dist, reduction='batchmean')
        
        # KL(Q || M)
        kl_q_m = F.kl_div(log_m_dist, q_dist, reduction='batchmean')

        l_js = 0.5 * (kl_p_m + kl_q_m)

        # l_kl = F.kl_div(q_dist.log(), p_dist, reduction='batchmean')
        # l_total = l_forward + (self.w_cycle * l_cycle) + (self.w_kl * l_kl)

        l_total = l_forward + (self.w_cycle * l_cycle) + (self.w_kl * l_js)

        self.log_dict({
            "train_loss_total": l_total,
            "train_loss_forward": l_forward,
            "train_loss_cycle": l_cycle,
            # "train_loss_kl": l_kl
            "train_loss_js": l_js
        }, prog_bar=True, logger=True, sync_dist=True)

        return l_total

# vim: ts=4 sw=4 expandtab

# from email.generator import Generator
import math
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from molbart.models import _AbsTransformerModel
from molbart.models.util import PreNormDecoderLayer, PreNormEncoderLayer

# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------- Pre-train Models --------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class BARTModel(_AbsTransformerModel):
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
        schedule="cycle",
        warm_up_steps=None,
        optimizer="adam",
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(
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
            schedule,
            warm_up_steps,
            optimizer,
            dropout,
            **kwargs,
        )

        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"

        self.encoder = nn.TransformerEncoder(
            PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.decoder = nn.TransformerDecoder(
            PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.loss_function = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)

        self.token_fc = nn.Linear(d_model, vocabulary_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        """Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        encoder_input = x["encoder_input"]
        decoder_input = x["decoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
        decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)
        decoder_embeddings = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embeddings.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=encoder_embs.device)

        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        model_output = self.decoder(
            decoder_embeddings,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=encoder_pad_mask.clone(),
        )

        token_output = self.token_fc(model_output)

        output = {"model_output": model_output, "token_output": token_output}

        return output

    def encode(self, batch):
        """Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)
        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, batch):
        """Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embeddings = self._construct_input(decoder_input)

        sequence_length, _, _ = tuple(decoder_embeddings.size())
        tgt_mask = self._generate_square_subsequent_mask(sequence_length, device=decoder_embeddings.device)

        decoder_output = self.decoder(
            decoder_embeddings,
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask,
        )
        token_log_probabilities = self.generator(decoder_output)
        return token_log_probabilities

    def generator(self, decoder_output):
        token_log_probabilities = self.log_softmax(self.token_fc(decoder_output))
        return token_log_probabilities

    def _calc_loss(self, batch_input, model_output):
        """Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocabulary_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_function(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy", return_tokenized=False):
        """Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        if hasattr(self.sampler, "sample_molecules"):
            mol_strs, log_lhs = self.sampler.sample_molecules(
                self,
                batch_input,
                self.num_beams,
                sampling_alg,
                return_tokenized=return_tokenized,
            )
        else:
            enc_input = batch_input["encoder_input"]
            enc_mask = batch_input["encoder_pad_mask"]
            encode_input = {"encoder_input": enc_input, "encoder_pad_mask": enc_mask}
            memory = self.encode(encode_input)
            mem_mask = enc_mask.clone()

            _, batch_size, _ = tuple(memory.size())

            decode_fn = partial(self._decode_fn, memory=memory, mem_pad_mask=mem_mask)

            if sampling_alg == "greedy":
                mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size, memory.device)

            elif sampling_alg == "beam":
                mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, memory.device, k=self.num_beams)

            else:
                raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask,
        }
        model_output = self.decode(decode_input)
        return model_output

    def decode_batch(self, batch, return_last=True):
        """Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """
        decoder_input = batch["decoder_input"].transpose(0, 1)
        memory_input = batch["memory_input"].permute(1, 0, 2)
        memory_pad_mask = batch["memory_pad_mask"]

        decoder_embeddings = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embeddings.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_embeddings.device).to(
            decoder_embeddings.device
        )

        decoder_output = self.decoder(
            decoder_embeddings,
            memory_input,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask,
        )

        token_probabilities = self.generator(decoder_output)
        if return_last:
            return token_probabilities[-1, :, :]
        else:
            return token_probabilities


class UnifiedModel(_AbsTransformerModel):
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
        schedule="cycle",
        warm_up_steps=None,
        optimizer="adam",
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(
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
            schedule,
            warm_up_steps,
            dropout,
            **kwargs,
        )

        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        self.token_fc = nn.Linear(d_model, vocabulary_size)
        self.loss_function = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        """Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        enc_input = x["encoder_input"]
        enc_mask = x["encoder_pad_mask"]
        dec_input = x["decoder_input"]
        dec_mask = x["decoder_pad_mask"]
        att_mask = x["attention_mask"]

        model_input = torch.cat((enc_input, dec_input), dim=0)
        pad_mask = torch.cat((enc_mask, dec_mask), dim=0).transpose(0, 1)
        embs = self._construct_input(model_input)

        model_output = self.encoder(embs, mask=att_mask, src_key_padding_mask=pad_mask)
        token_output = self.token_fc(model_output)

        output = {"model_output": model_output, "token_output": token_output}

        return output

    def _calc_loss(self, batch_input, model_output):
        """Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        tgt_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, tgt_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocabulary_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size, _ = tuple(token_output.size())
        tgt_len, tgt_batch_size = tuple(target.size())

        assert seq_len == tgt_len
        assert batch_size == tgt_batch_size

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_function(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~target_mask
        num_tokens = inv_target_mask.sum()

        loss = loss * inv_target_mask
        loss = loss.sum() / num_tokens

        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_token_ids = batch_input["encoder_input"]
        enc_pad_mask = batch_input["encoder_pad_mask"]

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        enc_seq_len, batch_size = tuple(enc_token_ids.size())
        self.sampler.max_seq_len = self.max_seq_len - enc_seq_len

        decode_fn = partial(self._decode_fn, enc_token_ids=enc_token_ids, enc_pad_mask=enc_pad_mask)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size, enc_token_ids.device)

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, enc_token_ids.device, k=self.num_beams)

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, enc_token_ids, enc_pad_mask):
        # Strip off the start token for the decoded sequence
        dec_token_ids = token_ids[1:, :]

        enc_length, _ = tuple(enc_token_ids.shape)
        dec_length, _ = tuple(dec_token_ids.shape)
        att_mask = self._build_att_mask(enc_length - 1, dec_length + 1, device=dec_token_ids.device)

        model_input = {
            "encoder_input": enc_token_ids,
            "encoder_pad_mask": enc_pad_mask,
            "decoder_input": dec_token_ids,
            "decoder_pad_mask": pad_mask[1:, :],
            "attention_mask": att_mask,
        }
        token_output = self.forward(model_input)["token_output"]
        token_probs = self.log_softmax(token_output)
        return token_probs

    def _build_att_mask(self, enc_length, dec_length, device="cpu"):
        seq_len = enc_length + dec_length
        enc_mask = torch.zeros((seq_len, enc_length), device=device)
        upper_dec_mask = torch.ones((enc_length, dec_length), device=device)
        lower_dec_mask = torch.ones((dec_length, dec_length), device=device).triu_(1)
        dec_mask = torch.cat((upper_dec_mask, lower_dec_mask), dim=0)
        mask = torch.cat((enc_mask, dec_mask), dim=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

class CycleConsistencyBARTModel(BARTModel):
    def __init__(self, *args, **kwargs):
        self.w_retro = kwargs.pop("w_retro", 1.0)
        self.w_kl = kwargs.pop("w_kl", 0.1)

        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """
        Overrides the original training_step to implement the composite loss.
        L_total = L_forward + w_retro * L_retro + w_kl * L_KL
        """

        forward_output = self.forward(batch)
        l_forward = self._calc_loss(batch, forward_output)


        with torch.no_grad():
            predicted_product_ids = torch.argmax(forward_output["token_output"], dim=-1)
            predicted_product_mask = batch["target_mask"].clone()

        retro_batch = {
            "encoder_input": predicted_product_ids,
            "encoder_pad_mask": predicted_product_mask,
            "decoder_input": batch["encoder_input"][:-1, :],
            "decoder_pad_mask": batch["encoder_pad_mask"][:-1, :],
            "target": batch["encoder_input"][1:, :],
            "target_mask": batch["encoder_pad_mask"][1:, :],
        }

        retro_output = self.forward(retro_batch)
        l_retro = self._calc_loss(retro_batch, retro_output)


        # Get the hidden state of the [EOS] token from the FORWARD pass decoder
        forward_target_ids = batch["target"]
        forward_eos_indices = (forward_target_ids == self.sampler.tokenizer["end"]) | (forward_target_ids == self.pad_token_idx)
        forward_eos_locs = forward_eos_indices.long().argmax(dim=0)
        forward_decoder_memory = forward_output["model_output"]
        product_eos_repr = forward_decoder_memory[forward_eos_locs, torch.arange(forward_decoder_memory.size(1))]

        # Get the hidden state of the [EOS] token from the RETRO pass decoder
        retro_target_ids = retro_batch["target"]
        retro_eos_indices = (retro_target_ids == self.sampler.tokenizer["end"]) | (retro_target_ids == self.pad_token_idx)
        retro_eos_locs = retro_eos_indices.long().argmax(dim=0)
        retro_decoder_memory = retro_output["model_output"]
        reactant_eos_repr = retro_decoder_memory[retro_eos_locs, torch.arange(retro_decoder_memory.size(1))]

        l_kl = F.kl_div(
            F.log_softmax(reactant_eos_repr, dim=-1),
            F.softmax(product_eos_repr.detach(), dim=-1),
            reduction='batchmean'
        )

        l_total = l_forward + (self.w_retro * l_retro) + (self.w_kl * l_kl)

        self.log_dict({
            "train_loss_total": l_total,
            "train_loss_forward": l_forward,
            "train_loss_retro": l_retro,
            "train_loss_kl": l_kl
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
        token_embeddings = self.src_embed.weight
        
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

        batch_size = forward_decoder_memory.size(1)

        forward_decoder_memory = forward_output["model_output"]
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

import torch
import torch.utils.data as tud

from molbart.models.base_transformer import _AbsTransformerModel
from typing import Any, Dict, List, Tuple

NEG_INF = -1e20

class Node:
    def __init__(self, model, x, vocabulary, device, data_device="cpu", batch_size=64):
        """
        Initialize a Node used for autoregression
        predictions, such as greedy search, multinomial
        sampling, or beam search

        Parameters:
            model (Any): any autoregressive model
            x (tuple(torch.tensor,)): a torch tensor representing
                              additional data to pass to
                              the regression model
            vocabulary (Vocabulary): a vocabulary object
            device (torch.device, str): device where to place
                                        the model
            data_device (torch.device, str): device where to place
                                             the data. WARNING! Use
                                             gpu here sparingly.
            batch_size(int): internal batch size used for the beam search
                             (Default: 64)

        """
        assert isinstance(device, torch.device) or isinstance(device, str)
        assert isinstance(data_device, torch.device) or isinstance(data_device, str)

        if isinstance(device, str):
            device = torch.device(device)

        if isinstance(data_device, str):
            data_device = torch.device(data_device)

        self.model = model
        self.device = device
        self.data_device = data_device

        src = x["encoder_input"]
        src_mask = x["encoder_pad_mask"]
        self.batch_size = batch_size  # min(batch_size, len(src))

        with torch.no_grad():
            self.model = self.model.eval()

            if next(self.model.parameters()).device != self.device:
                self.model = self.model.to(self.device)
            if src.device != self.device:
                src = src.to(self.device)
            if src_mask.device != self.device:
                src_mask = src_mask.to(self.device)

            self.memory = self.model.encode(x).detach().permute(1, 0, 2)
            self.memory_mask = src_mask.detach().transpose(0, 1)

            if self.memory.device != self.data_device:
                self.memory = self.memory.to(self.data_device)
            if self.memory_mask.device != self.data_device:
                self.memory_mask = self.memory_mask.to(self.data_device)

        self.vocabulary = vocabulary

        self.seq = torch.ones((self.memory.shape[0], 1), dtype=torch.long) * self.vocabulary["start"]
        self.seq = self.seq.detach()

        if self.seq.device != self.data_device:
            self.seq = self.seq.to(self.data_device)

        self.ll_mask = torch.tensor([False])
        self.pos = 0

    def set_beam_width(self, beam_width):
        self.beam_width = beam_width

    def _get_topk(
        self, loglikelihoods: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the top-k log-likelihoods and their corresponding token indices.
        Pads if the vocabulary size is smaller than the beam width.
        """
        vocab_size = loglikelihoods.shape[-1]
        k = min(vocab_size, self.beam_width)

        top_loglikelihoods, top_tokens = loglikelihoods.topk(k=k, dim=-1)

        if vocab_size < self.beam_width:
            padding_needed = self.beam_width - vocab_size
            
            pad_loglikelihoods = torch.full(
                (len(loglikelihoods), padding_needed),
                NEG_INF,
                dtype=top_loglikelihoods.dtype,
                device=top_loglikelihoods.device,
            )

            pad_tokens = torch.zeros(
                (len(top_tokens), padding_needed),
                dtype=top_tokens.dtype,
                device=top_tokens.device,
            )
            top_loglikelihoods = torch.cat([top_loglikelihoods, pad_loglikelihoods], dim=-1)
            top_tokens = torch.cat([top_tokens, pad_tokens], dim=-1)

        return top_loglikelihoods, top_tokens

    def _init_action(self, loglikelihood):
        # Perform the first step
        top_loglikelihoods, next_tokens = self._get_topk(loglikelihood)

        self.loglikelihood = top_loglikelihoods.view(-1, 1)
        next_tokens = next_tokens.view(-1, 1)

        # (bsz, seq) -> (bsz, 1, seq) -> (bsz , beam, seq) -> (bsz * beam, seq)
        self.seq = self.seq.unsqueeze(1).repeat(1, self.beam_width, 1)
        self.seq = self.seq.view(-1, self.seq.shape[-1])

        self.memory = self.memory.unsqueeze(1).repeat(1, self.beam_width, 1, 1)
        self.memory = self.memory.view(-1, *self.memory.shape[2:])

        self.memory_mask = self.memory_mask.unsqueeze(1).repeat(1, self.beam_width, 1)
        self.memory_mask = self.memory_mask.view(-1, self.memory_mask.shape[-1])

        self.seq = torch.cat((self.seq, next_tokens), dim=-1)

        # VERY IMPORTANT! we need a mask for
        # the log likelihood when reaching the eos
        # self.ll_mask = torch.zeros(len(self.loglikelihood), dtype=torch.bool)
        self.ll_mask = torch.any(self.seq == self.vocabulary["end"], dim=-1)

    def get_actions(self):
        batch_size = self.batch_size
        next_loglikelihood = []

        local_dataset = tud.TensorDataset(self.memory, self.memory_mask, self.seq)
        local_loader = tud.DataLoader(local_dataset, batch_size=batch_size)

        # make sure that the local_loader
        # will be iterated over only once
        iterator = iter(local_loader)

        with torch.no_grad():
            for memory, memory_mask, src in local_loader:
                if memory.device != self.device:
                    memory = memory.to(self.device)

                if memory_mask.device != self.device:
                    memory_mask = memory_mask.to(self.device)

                if src.device != self.device:
                    src = src.to(self.device)

                # memory and src in Node is currently saved
                # with bsz as the first element
                src_transposed = src.transpose(0, 1)
                memory_permuted = memory.permute(1, 0, 2)
                memory_mask_transposed = memory_mask.transpose(0, 1)

                batch = {
                    "decoder_input": src_transposed,
                    "memory_input": memory_permuted,
                    "memory_pad_mask": memory_mask_transposed,
                }

                ll, _ = self.model.decode(batch, return_last=True)
                next_loglikelihood.append(ll)
        next_loglikelihood = torch.cat(next_loglikelihood, axis=0)
        next_loglikelihood = next_loglikelihood.detach()
        if next_loglikelihood != self.data_device:
            next_loglikelihood = next_loglikelihood.to(self.data_device)

        return next_loglikelihood

    def action(self, next_loglikelhihood):
        if self.pos == 0:
            self._init_action(next_loglikelhihood)
        else:
            vocabulary_size = len(self.vocabulary)
            # set loglikehihood to the maxium (0)
            # when observed an eos_token
            next_loglikelhihood[self.ll_mask, :] = (
                torch.minimum(self.loglikelihood.min(), next_loglikelhihood.min()) - 1.0
            )
            next_loglikelhihood[self.ll_mask, self.vocabulary["end"]] = 0.0
            # done

            ll = (self.loglikelihood + next_loglikelhihood).view(-1, self.beam_width, vocabulary_size)
            ll, idx = self._get_topk(ll.flatten(start_dim=1))

            # tricky indexing
            next_chars = torch.remainder(idx, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            if best_candidates.device != self.device:
                best_candidates = best_candidates.to(self.device)
            # done

            src = self.seq.view(-1, self.beam_width, self.seq.shape[-1])
            i = torch.arange(len(src)).unsqueeze(-1).repeat(1, self.beam_width).flatten()
            j = best_candidates.flatten()
            self.seq = src[i, j].view(-1, self.seq.shape[-1])

            self.seq = torch.cat((self.seq, next_chars), dim=-1)
            self.loglikelihood = ll.view(-1, 1)

            # update ll mask
            self.ll_mask = torch.any(self.seq == self.vocabulary["end"], dim=-1)
        self.pos = self.pos + 1

    @staticmethod
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        A = subsequent_mask == 0
        return A.type(torch.long)

class Criterion:
    def __call__(self, node):
        raise NotImplementedError("Not implemented")


class MaxLength(Criterion):
    def __init__(self, max_length):
        super(MaxLength, self).__init__()
        self.max_length = max_length

    def __call__(self, node):
        return node.pos >= self.max_length - 1


class EOS(Criterion):
    def __init__(self):
        super(EOS, self).__init__()

    def __call__(self, node):
        return torch.all(node.ll_mask).item()


class LogicalAnd(Criterion):
    def __init__(self, criteria):
        super(LogicalAnd, self).__init__()
        self.criteria = criteria

    def __call__(self, node):
        return all([c(node) for c in self.criteria])


class LogicalOr(Criterion):
    def __init__(self, criteria):
        super(LogicalOr, self).__init__()
        self.criteria = criteria

    def __call__(self, node):
        return any([c(node) for c in self.criteria])


def beamsearch(node, beamsize, stop_criterion):
    node.set_beam_width(beamsize)
    print("Sampling with beam size: " + str(beamsize))

    while not stop_criterion(node):
        if torch.all(node.ll_mask):
            print("All beams finished.")
            break

        next_loglikelihood = node.get_actions()
        node.action(next_loglikelihood)

    return node

# vim: ts=4 sw=4 expandtab

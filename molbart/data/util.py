""" Module containing helper routines for the DataModules """
from typing import Any, List, Optional, Tuple

import torch
from molbart.utils.tokenizers import ChemformerTokenizer, ListOfStrList, TokensMasker


class BatchEncoder:
    """
    Encodes a sequence for the Chemformer model

    This procedure includes:
        1. Tokenization
        2. Optional masking
        3. Padding
        4. Optional adding separation token to the end
        5. Checking of sequence lengths and possibly truncation
        6. Conversion to pytorch.Tensor

    Encoding is carried out by

    .. code-block::

        id_tensor, mask_tensor = encoder(batch, mask=True)

    where `batch` is a list of strings to be encoded and `mask` is
    a flag that can be used to toggled the masking.

    :param tokenizer: the tokenizer to use
    :param masker: the masker to use
    :param max_seq_len: the maximum allowed list length
    """

    def __init__(
        self,
        tokenizer: ChemformerTokenizer,
        masker: Optional[TokensMasker],
        max_seq_len: int,
    ):
        self._tokenizer = tokenizer
        self._masker = masker
        self._max_seq_len = max_seq_len

    def __call__(
        self,
        batch: List[str],
        mask: bool = False,
        add_sep_token: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokenizer.tokenize(batch)
        if mask and self._masker is not None:
            tokens, _ = self._masker(tokens)
        tokens, pad_mask = self._pad_seqs(tokens, self._tokenizer.special_tokens["pad"])

        if add_sep_token:
            sep_token = self._tokenizer.special_tokens["sep"]
            tokens = [itokens + [sep_token] for itokens in tokens]
            pad_mask = [imasks + [0] for imasks in pad_mask]

        tokens, pad_mask = self._check_seq_len(tokens, pad_mask)
        id_data = self._tokenizer.convert_tokens_to_ids(tokens)
        id_tensor = torch.stack(id_data).transpose(0, 1)
        mask_tensor = torch.tensor(pad_mask, dtype=torch.bool).transpose(0, 1)
        return id_tensor, mask_tensor

    def _check_seq_len(self, tokens: ListOfStrList, mask: List[List[int]]) -> Tuple[ListOfStrList, List[List[int]]]:
        """Warn user and shorten sequence if the tokens are too long, otherwise return original"""

        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self._max_seq_len:
            print(f"WARNING -- Sequence length {seq_len} is larger than maximum sequence size")

            tokens_short = [ts[: self._max_seq_len] for ts in tokens]
            mask_short = [ms[: self._max_seq_len] for ms in mask]

            return tokens_short, mask_short

        return tokens, mask

    @staticmethod
    def _pad_seqs(seqs: List[Any], pad_token: Any) -> Tuple[List[Any], List[int]]:
        pad_length = max([len(seq) for seq in seqs])
        padded = [seq + ([pad_token] * (pad_length - len(seq))) for seq in seqs]
        masks = [([0] * len(seq)) + ([1] * (pad_length - len(seq))) for seq in seqs]
        return padded, masks

    def merge_and_rebatch(
        self,
        id_tensor1: torch.Tensor,
        mask_tensor1: torch.Tensor,
        id_tensor2: torch.Tensor,
        mask_tensor2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merges two batches of encoded sequences using vectorized operations.

        The process for each pair of sequences is:
        1. De-pad the sequences.
        2. Remove the trailing EOS token.
        3. Concatenate seq1, a SEP token, and seq2.
        4. Append a new EOS token.
        5. Re-pad the new batch of merged sequences using torch.nn.pad_sequence.
        6. Truncate if longer than max_seq_len.

        :param id_tensor1: The first batch of token ID tensors (seq_len, batch_size).
        :param mask_tensor1: The first batch of padding mask tensors.
        :param id_tensor2: The second batch of token ID tensors.
        :param mask_tensor2: The second batch of padding mask tensors.
        :return: A tuple containing the new merged and padded ID and mask tensors.
        """

        device = id_tensor1.device
        if id_tensor2.device != device or mask_tensor1.device != device or mask_tensor2.device != device:
             raise ValueError("All input tensors must be on the same device.")

        pad_id = self._tokenizer.convert_tokens_to_ids([self._tokenizer.special_tokens["pad"]])[0].item()
        sep_id = self._tokenizer.convert_tokens_to_ids([self._tokenizer.special_tokens["sep"]])[0].item()
        eos_id = self._tokenizer.convert_tokens_to_ids([self._tokenizer.special_tokens["eos"]])[0].item()
        
        # Transpose to (batch_size, seq_len) for easier processing
        id_tensor1, mask_tensor1 = id_tensor1.transpose(0, 1), mask_tensor1.transpose(0, 1)
        id_tensor2, mask_tensor2 = id_tensor2.transpose(0, 1), mask_tensor2.transpose(0, 1)

        batch_size = id_tensor1.shape[0]

        # 1. Vectorized calculation of sequence lengths
        lengths1 = (~mask_tensor1).sum(dim=1)
        lengths2 = (~mask_tensor2).sum(dim=1)

        # 2. Vectorized removal of EOS token from lengths
        batch_indices = torch.arange(batch_size, device=device)
        
        # For batch 1
        last_token_indices1 = (lengths1 - 1).clamp(min=0)
        last_tokens1 = id_tensor1[batch_indices, last_token_indices1]
        is_eos1 = (last_tokens1 == eos_id) & (lengths1 > 0)
        new_lengths1 = lengths1 - is_eos1.long()

        # For batch 2
        last_token_indices2 = (lengths2 - 1).clamp(min=0)
        last_tokens2 = id_tensor2[batch_indices, last_token_indices2]
        is_eos2 = (last_tokens2 == eos_id) & (lengths2 > 0)
        new_lengths2 = lengths2 - is_eos2.long()

        sep_tensor = torch.tensor([sep_id], dtype=torch.long, device=device)
        eos_tensor = torch.tensor([eos_id], dtype=torch.long, device=device)

        merged_sequences = [
            torch.cat([
                id_tensor1[i, :new_lengths1[i]],
                sep_tensor,
                id_tensor2[i, :new_lengths2[i]],
                eos_tensor
            ]) for i in range(batch_size)
        ]

        # 4. Pad the new batch using optimized PyTorch utility
        padded_ids = pad_sequence(merged_sequences, batch_first=True, padding_value=pad_id)

        # 5. Check sequence length and truncate if necessary
        if padded_ids.shape[1] > self._max_seq_len:
            print(f"WARNING -- Merged sequence length {padded_ids.shape[1]} is > {self._max_seq_len}, truncating.")
            padded_ids = padded_ids[:, :self._max_seq_len]

        # 6. Create the new padding mask and transpose to (seq_len, batch_size)
        final_mask = (padded_ids == pad_id)
        
        return padded_ids.transpose(0, 1), final_mask.transpose(0, 1)


def build_attention_mask(enc_length: int, dec_length: int) -> torch.Tensor:
    """
    Building the attention mask for the unified model

    :param enc_length: the length of the encoder
    :param dec_length: the length of the decoder
    :return: the mask tensor
    """
    seq_len = enc_length + dec_length
    enc_mask = torch.zeros((seq_len, enc_length))
    upper_dec_mask = torch.ones((enc_length, dec_length))
    lower_dec_mask = torch.ones((dec_length, dec_length)).triu_(1)
    dec_mask = torch.cat((upper_dec_mask, lower_dec_mask), dim=0)
    mask = torch.cat((enc_mask, dec_mask), dim=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


def build_target_mask(enc_length: int, dec_length: int, batch_size: int) -> torch.Tensor:
    """
    Build the target mask for the unified model

    :param enc_length: the length of the encoder
    :param dec_length: the length of the decoder
    :param batch_size: the batch size
    :return: the mask tensor
    """
    # Take one and add one because we shift the target left one token
    # So the first token of the target output will be at the same position as the separator token of the input,
    # And the separator token is not present in the output
    enc_mask = [1] * (enc_length - 1)
    dec_mask = [0] * (dec_length + 1)
    mask = [enc_mask + dec_mask] * batch_size
    mask = torch.tensor(mask, dtype=torch.bool).T
    return mask

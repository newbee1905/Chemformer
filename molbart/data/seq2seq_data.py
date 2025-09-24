""" Module containing classes to load seq2seq data"""
import os
import sys
import pandas as pd
from rdkit import Chem
from typing import Any, Dict, List, Tuple

from molbart.data.base import ReactionListDataModule, _setup_lmdb_datasets


class Uspto50DataModule(ReactionListDataModule):
    """
    DataModule for the USPTO-50 dataset

    The reactions as well as a type token are read from
    a pickled DataFrame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._include_type_token = kwargs.get("include_type_token", False)

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        reactants = [Chem.MolToSmiles(item["reactants_mol"]) for item in batch]
        products = [Chem.MolToSmiles(item["products_mol"]) for item in batch]

        if train:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)

        if self._include_type_token and not self.reverse:
            reactants = [item["reaction_type"] + smi for item, smi in zip(batch, reactants)]
        if self._include_type_token and self.reverse:
            products = [item["reaction_type"] + smi for item, smi in zip(batch, products)]

        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants_mol": df["reactants_mol"].tolist(),
            "products_mol": df["products_mol"].tolist(),
            "reaction_type": df["reaction_type"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)


class UsptoMixedDataModule(ReactionListDataModule):
    """
    DataModule for the USPTO-Mixed dataset

    The reactions are read from a pickled DataFrame
    """

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        reactants = [Chem.MolToSmiles(item["reactants_mol"]) for item in batch]
        products = [Chem.MolToSmiles(item["products_mol"]) for item in batch]
        if train:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)
        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants_mol": df["reactants_mol"].tolist(),
            "products_mol": df["products_mol"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)


class UsptoSepDataModule(ReactionListDataModule):
    """
    DataModule for the USPTO-Separated dataset

    The reactants, reagents and products are read from
    a pickled DataFrame
    """

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        reactants = [Chem.MolToSmiles(item["reactants_mol"]) for item in batch]
        reagents = [Chem.MolToSmiles(item["reagents_mol"]) for item in batch]
        products = [Chem.MolToSmiles(item["products_mol"]) for item in batch]

        if train:
            reactants = self._batch_augmenter(reactants)
            reagents = self._batch_augmenter(reagents)
            products = self._batch_augmenter(products)

        reactants = [react_smi + ">" + reag_smi for react_smi, reag_smi in zip(reactants, reagents)]

        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants_mol": df["reactants_mol"].tolist(),
            "products_mol": df["products_mol"].tolist(),
            "reagents_mol": df["reagents_mol"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)

class UsptoCycleDataModule(ReactionListDataModule):
    """
    DataModule for the USPTO-Separated dataset

    The reactants, reagents and products are read from
    a pickled DataFrame
    """

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        r_smi = [Chem.MolToSmiles(item["reactants_mol"]) for item in batch]
        g_smi = [Chem.MolToSmiles(item["reagents_mol"]) for item in batch]
        p_smi = [Chem.MolToSmiles(item["products_mol"]) for item in batch]

        if train:
            sys.stdout = open(os.devnull, 'w')
            r_smi = self._batch_augmenter(r_smi)
            g_smi = self._batch_augmenter(g_smi)
            p_smi = self._batch_augmenter(p_smi)
            sys.stdout = sys.__stdout__

        # reactants = [react_smi + "<SEP>" + reag_smi for react_smi, reag_smi in zip(reactants, reagents)]
        rg_smi = [r + "<SEP>" + g for r, g in zip(r_smi, g_smi)]
        pg_smi = [p + "<SEP>" + g for p, g in zip(p_smi, g_smi)]

        return rg_smi, p_smi, r_smi, pg_smi

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants_mol": df["reactants_mol"].tolist(),
            "products_mol": df["products_mol"].tolist(),
            "reagents_mol": df["reagents_mol"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)

    def _collate(self, batch: List[Dict[str, Any]], train: bool = True) -> Dict[str, Any]:
        rg_smi, p_smi, r_smi, pg_smi = self._get_sequences(batch, train)

        encoder_ids, encoder_mask = self._encoder(rg_smi)
        decoder_ids, decoder_mask = self._encoder(p_smi)

        reactants_ids, reactants_mask = self._encoder(r_smi)
        retro_ids, retro_mask = self._encoder(pg_smi)

        product_lengths = decoder_mask.sum(dim=0).long()
        reactant_lengths = reactants_mask.sum(dim=0).long()

        return {
            # Main Task
            "encoder_input": encoder_ids,
            "encoder_pad_mask": encoder_mask,
            "decoder_input": decoder_ids[:-1, :],
            "decoder_pad_mask": decoder_mask[:-1, :],
            "target": decoder_ids.clone()[1:, :],
            "target_mask": decoder_mask.clone()[1:, :],
            "target_smiles": p_smi,
            
            # Cycle Task components
            "reactants_ids": reactants_ids,
            "reactants_mask": reactants_mask,
            "retro_ids": retro_ids,
            "retro_mask": retro_mask,

            "product_lengths": product_lengths,
            "reactant_lengths": reactant_lengths,
        }


class MolecularOptimizationDataModule(ReactionListDataModule):
    """
    DataModule for a dataset for molecular optimization

    The input and ouput molecules, as well as a the property
    tokens are read from a pickled DataFrame
    """

    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        input_smiles = [Chem.MolToSmiles(item["input_mols"]) for item in batch]
        output_smiles = [Chem.MolToSmiles(item["output_mols"]) for item in batch]

        if train:
            input_smiles = self._batch_augmenter(input_smiles)
            output_smiles = self._batch_augmenter(output_smiles)

        input_smiles = [item["prop_tokens"] + smi for item, smi in zip(batch, input_smiles)]

        return input_smiles, output_smiles

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "prop_tokens": df["property_tokens"].tolist(),
            "input_mols": df["input_mols"].tolist(),
            "output_mols": df["output_mols"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)

class Uspto50DataModuleLMDB(Uspto50DataModule):
    """DataModule for the USPTO-50 dataset from an LMDB database."""
    def setup(self, stage: str = None):
        """Loads indices from LMDB and creates train/val/test datasets."""
        _setup_lmdb_datasets(self)


class UsptoMixedDataModuleLMDB(UsptoMixedDataModule):
    """DataModule for the USPTO-Mixed dataset from an LMDB database."""
    def setup(self, stage: str = None):
        _setup_lmdb_datasets(self)


class UsptoSepDataModuleLMDB(UsptoSepDataModule):
    """DataModule for the USPTO-Separated dataset from an LMDB database."""
    def setup(self, stage: str = None):
        _setup_lmdb_datasets(self)

class UsptoCycleDataModuleLMDB(UsptoCycleDataModule):
    """DataModule for the USPTO-Separated dataset from an LMDB database."""
    def setup(self, stage: str = None):
        _setup_lmdb_datasets(self)

class MolecularOptimizationDataModuleLMDB(MolecularOptimizationDataModule):
    """DataModule for molecular optimization from an LMDB database."""
    def setup(self, stage: str = None):
        _setup_lmdb_datasets(self)

# vim: ts=4 sw=4 expandtab

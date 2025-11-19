""" Module containing classes to load seq2seq data"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem 
from rdkit.Chem import AllChem
from typing import Any, Dict, List, Tuple, Set

from molbart.data.base import ReactionListDataModule, _setup_lmdb_datasets

from ase import Atoms
import molbart.utils.graph_properties as structure


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


# class UsptoSepDataModule(ReactionListDataModule):
#     """
#     DataModule for the USPTO-Separated dataset

#     The reactants, reagents and products are read from
#     a pickled DataFrame
#     """

#     def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
#         reactants = [Chem.MolToSmiles(item["reactants_mol"]) for item in batch]
#         reagents = [Chem.MolToSmiles(item["reagents_mol"]) for item in batch]
#         products = [Chem.MolToSmiles(item["products_mol"]) for item in batch]

#         if train:
#             sys.stdout = open(os.devnull, 'w')
#             reactants = self._batch_augmenter(reactants)
#             reagents = self._batch_augmenter(reagents)
#             products = self._batch_augmenter(products)
#             sys.stdout = sys.__stdout__

#         reactants = [react_smi + ">" + reag_smi for react_smi, reag_smi in zip(reactants, reagents)]

#         return reactants, products

#     def _load_all_data(self) -> None:
#         df = pd.read_pickle(self.dataset_path).reset_index()
#         self._all_data = {
#             "reactants_mol": df["reactants_mol"].tolist(),
#             "products_mol": df["products_mol"].tolist(),
#             "reagents_mol": df["reagents_mol"].tolist(),
#         }
#         self._set_split_indices_from_dataframe(df)

class UsptoSepDataModule(ReactionListDataModule):
    """
    DataModule for the USPTO-Separated dataset

    The reactants, reagents and products are read from
    a pickled DataFrame.

    This class is modified to generate 3D graph inputs
    for both encoder (reactants+reagents) and decoder (products).
    """

    def _atom_features(self, atom: Chem.Atom) -> np.ndarray:
        """Extracts features for a single atom."""

        # Simple features: Atomic Num, Formal Charge, IsAromatic, TotalNumHs
        # You can expand this (e.g., one-hot hybridization, ring status, etc.)
        features = [
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
        ]

        return np.array(features, dtype=np.float32)

    def _bond_features(self, bond: Chem.Bond) -> np.ndarray:
        """Extracts features for a single bond."""

        bond_type_map = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 2.0,
            Chem.BondType.TRIPLE: 3.0,
            Chem.BondType.AROMATIC: 1.5,
        }

        features = [
            bond_type_map.get(bond.GetBondType(), 0.0),
            bond.GetIsAromatic(),
        ]

        return np.array(features, dtype=np.float32)

    def _build_graph_batch(self, smiles_list: List[str]) -> Dict[str, torch.Tensor]:
        """
        Generates and collates a batch of graphs from SMILES strings.
        Attempts 3D embedding, falls back to 2D.
        """
        try:
            node_features_list = []
            edge_indices_list = []
            edge_features_list = []
            positions_list = []
            batch_indices = []
            node_offset = 0

            for i, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # Generate 3D coordinates
                mol_h = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = 0
                embed_status = AllChem.EmbedMolecule(mol_h, params)
                
                use_3d = True
                if embed_status == -1:
                    # 3D embedding failed, fall back to 2D
                    AllChem.Compute2DCoords(mol_h)
                    use_3d = False
                else:
                    try:
                        AllChem.UFFOptimizeMolecule(mol_h)
                    except Exception:
                        pass
                
                conf = mol_h.GetConformer()
                
                # Atom (Node) features and positions
                mol_node_features = []
                mol_positions = []
                for atom in mol_h.GetAtoms():
                    mol_node_features.append(self._atom_features(atom))
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    if use_3d:
                        mol_positions.append([pos.x, pos.y, pos.z])
                    else:
                        mol_positions.append([pos.x, pos.y, 0.0]) # Add z=0 for 2D

                node_features_list.append(np.array(mol_node_features))
                positions_list.append(np.array(mol_positions))

                # Bond (Edge) features
                mol_edge_indices = []
                mol_edge_features = []
                for bond in mol_h.GetBonds():
                    i_idx = bond.GetBeginAtomIdx()
                    j_idx = bond.GetEndAtomIdx()
                    
                    # Add edges in both directions
                    mol_edge_indices.extend([[i_idx, j_idx], [j_idx, i_idx]])
                    
                    bond_feats = self._bond_features(bond)
                    mol_edge_features.extend([bond_feats, bond_feats])

                # Apply offset for collated graph
                if mol_edge_indices:
                    edge_indices_list.append(np.array(mol_edge_indices) + node_offset)
                    edge_features_list.append(np.array(mol_edge_features))

                batch_indices.extend([i] * len(mol_node_features))
                node_offset += len(mol_node_features)

            # Collate into tensors
            batch_dict = {
                "x": torch.from_numpy(np.concatenate(node_features_list, axis=0)).float() if node_features_list else torch.empty(0, dtype=torch.float32),
                "edge_index": torch.from_numpy(np.concatenate(edge_indices_list, axis=0)).t().contiguous().long() if edge_indices_list else torch.empty((2, 0), dtype=torch.long),
                "edge_attr": torch.from_numpy(np.concatenate(edge_features_list, axis=0)).float() if edge_features_list else torch.empty(0, dtype=torch.float32),
                "pos": torch.from_numpy(np.concatenate(positions_list, axis=0)).float() if positions_list else torch.empty(0, dtype=torch.float32),
                "batch": torch.tensor(batch_indices, dtype=torch.long)
            }
            return batch_dict

        except Exception as e:
            print(e)


    def _get_sequences(self, batch: List[Dict[str, Any]], train: bool) -> Tuple[List[str], List[str]]:
        reactants = [Chem.MolToSmiles(item["reactants_mol"]) for item in batch]
        reagents = [Chem.MolToSmiles(item["reagents_mol"]) for item in batch]
        products = [Chem.MolToSmiles(item["products_mol"]) for item in batch]

        if train:
            sys.stdout = open(os.devnull, 'w')  # Silences RDKit warnings
            reactants = self._batch_augmenter(reactants)
            reagents = self._batch_augmenter(reagents)
            products = self._batch_augmenter(products)
            sys.stdout = sys.__stdout__

        # Combine reactants and reagents with '.' as requested
        graph_reactants = [react_smi + "." + reag_smi for react_smi, reag_smi in zip(reactants, reagents)]
        reactants = [react_smi + ">" + reag_smi for react_smi, reag_smi in zip(reactants, reagents)]

        return reactants, graph_reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants_mol": df["reactants_mol"].tolist(),
            "products_mol": df["products_mol"].tolist(),
            "reagents_mol": df["reagents_mol"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)

    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Overrides parent method to also generate graph batches.
        """
        encoder_smiles, graph_smiles, decoder_smiles = self._get_sequences(batch, train)

        # Tokenize for Transformer
        encoder_ids, encoder_mask = self._encoder(encoder_smiles, add_sep_token=self.unified_model and not self.reverse)
        decoder_ids, decoder_mask = self._encoder(decoder_smiles, add_sep_token=self.unified_model and self.reverse)

        # Build Graph Batches
        enc_graph_batch = self._build_graph_batch(graph_smiles)
        dec_graph_batch = self._build_graph_batch(decoder_smiles)

        if not self.reverse:
            return encoder_ids, encoder_mask, decoder_ids, decoder_mask, decoder_smiles, enc_graph_batch, dec_graph_batch

        return decoder_ids, decoder_mask, encoder_ids, encoder_mask, encoder_smiles, dec_graph_batch, enc_graph_batch

    def _collate(self, batch: List[Dict[str, Any]], train: bool = True) -> Dict[str, Any]:
        """
        Overrides base collate function to handle the new graph data
        returned by the overridden _transform_batch.
        """
        (
            encoder_ids,
            encoder_mask,
            decoder_ids,
            decoder_mask,
            smiles,
            enc_graph_batch,
            dec_graph_batch,
        ) = self._transform_batch(batch, train)

        if self.unified_model:
            model_batch = self._make_unified_model_batch(encoder_ids, encoder_mask, decoder_ids, decoder_mask, smiles)
        else:
            model_batch = {
                "encoder_input": encoder_ids,
                "encoder_pad_mask": encoder_mask,
                "decoder_input": decoder_ids[:-1, :],
                "decoder_pad_mask": decoder_mask[:-1, :],
                "target": decoder_ids.clone()[1:, :],
                "target_mask": decoder_mask.clone()[1:, :],
                "target_smiles": smiles,
            }

        model_batch["encoder_graph"] = enc_graph_batch
        model_batch["decoder_graph"] = dec_graph_batch
        
        return model_batch

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

        sys.stdout = open(os.devnull, 'w')
        r_smi = self._batch_augmenter(r_smi)
        g_smi = self._batch_augmenter(g_smi)
        p_smi = self._batch_augmenter(p_smi)
        sys.stdout = sys.__stdout__

        # reactants = [react_smi + "<SEP>" + reag_smi for react_smi, reag_smi in zip(reactants, reagents)]
        # rg_smi = [r + "<SEP>" + g for r, g in zip(r_smi, g_smi)]
        # pg_smi = [p + "<SEP>" + g for p, g in zip(p_smi, g_smi)]

        # return rg_smi, p_smi, r_smi, pg_smi

        return r_smi, g_smi, p_smi

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants_mol": df["reactants_mol"].tolist(),
            "products_mol": df["products_mol"].tolist(),
            "reagents_mol": df["reagents_mol"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)

    def _collate(self, batch: List[Dict[str, Any]], train: bool = True) -> Dict[str, Any]:
        # rg_smi, p_smi, r_smi, pg_smi = self._get_sequences(batch, train)

        # encoder_ids, encoder_mask = self._encoder(rg_smi)
        # decoder_ids, decoder_mask = self._encoder(p_smi)

        # reactants_ids, reactants_mask = self._encoder(r_smi)
        # retro_ids, retro_mask = self._encoder(pg_smi)

        # product_lengths = decoder_mask.sum(dim=0).long()
        # reactant_lengths = reactants_mask.sum(dim=0).long()

        # return {
        #     # Main Task
        #     "encoder_input": encoder_ids,
        #     "encoder_pad_mask": encoder_mask,
        #     "decoder_input": decoder_ids[:-1, :],
        #     "decoder_pad_mask": decoder_mask[:-1, :],
        #     "target": decoder_ids.clone()[1:, :],
        #     "target_mask": decoder_mask.clone()[1:, :],
        #     "target_smiles": p_smi,
            
        #     # Cycle Task components
        #     "reactants_ids": reactants_ids,
        #     "reactants_mask": reactants_mask,
        #     "retro_ids": retro_ids,
        #     "retro_mask": retro_mask,

        #     "product_lengths": product_lengths,
        #     "reactant_lengths": reactant_lengths,
        # }
        r_smi, g_smi, p_smi = self._get_sequences(batch, train)

        reactants_ids, reactants_mask = self._encoder(r_smi)
        products_ids, products_mask = self._encoder(p_smi)
        reagents_ids, reagents_mask = self._encoder(g_smi)

        return {
            # Main Task
            "encoder_input": reactants_ids,
            "encoder_pad_mask": reactants_mask,

            "decoder_input": products_ids[:-1, :],
            "decoder_pad_mask": products_mask[:-1, :],

            "target": products_ids.clone()[1:, :],
            "target_mask": products_mask.clone()[1:, :],

            "sub_encoder_input": reagents_ids,
            "sub_encoder_pad_mask": reagents_mask,
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

def get_3d_mol(smiles: str):
    """Generates an RDKit Mol object with 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
            
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if status == -1:
        print(f"Warning: Could not generate 3D conformer for {smiles}. Using random coords.")
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Chem.rdchem.AtomValenceException:
        print(f"Warning: Skipping UFF optimization due to valence error for {smiles}")

    conformer = mol.GetConformer()
    return mol, conformer

class NeighborList:
    def __init__(self, idx_i, idx_j):
        self.idx_i = idx_i
        self.idx_j = idx_j

class AtomsConverter:
    """Helper to get neighbor lists for schnetpack format."""
    def __init__(self, device="cpu", cutoff=5.0):
        self.device = device
        self.cutoff = cutoff

    def _get_neighbors(self, atoms):
        positions = torch.tensor(atoms.positions, device=self.device, dtype=torch.float)
        n_atoms = len(atoms)
        
        dist_matrix = torch.norm(
            positions.unsqueeze(1) - positions.unsqueeze(0), 
            dim=-1
        )
        
        adj = (dist_matrix > 0) & (dist_matrix < self.cutoff)
        idx_i, idx_j = torch.where(adj)
        
        return NeighborList(idx_i, idx_j)

def process_hybrid(
    token_ids: torch.Tensor, 
    smiles_string: str, 
    atom_token_ids: set,
    atoms_converter: AtomsConverter
):
    """
    Processes a SMILES string for a hybrid token/graph model.
    """
    
    # 1. Get molecule WITHOUT hydrogens for token matching
    mol_heavy = Chem.MolFromSmiles(smiles_string)
    if not mol_heavy:
        raise ValueError(f"RDKit could not parse SMILES: {smiles_string}")
    n_heavy_atoms = mol_heavy.GetNumAtoms()

    # 2. Run your tokenizing logic
    atom_mask_tokens = []
    atom_index_counter = 0
    
    token_ids_list = token_ids.tolist()

    for token_id in token_ids_list:
        if token_id in atom_token_ids:
            atom_mask_tokens.append(1)
            atom_index_counter += 1
        else:
            atom_mask_tokens.append(0)
    
    # 3. Check for mismatch
    if atom_index_counter != n_heavy_atoms:
        print(
            f"CRITICAL MASKING ERROR in '{smiles_string}': "
            f"Found {atom_index_counter} atom tokens, but RDKit found {n_heavy_atoms} HEAVY atoms."
        )
        return None

    # 4. Get 3D data WITH hydrogens for the GNN
    try:
        mol_with_H, conformer = get_3d_mol(smiles_string) 
    except Exception as e:
        print(f"Skipping bad SMILES during 3D generation: {e}")
        return None 

    n_total_atoms = mol_with_H.GetNumAtoms()
    
    # 5. Extract GNN data
    atomic_numbers = [atom.GetAtomicNum() for atom in mol_with_H.GetAtoms()]
    atom_positions = conformer.GetPositions()

    # 6. Create GNN heavy atom mask
    gnn_is_heavy_atom_mask = torch.zeros(n_total_atoms, dtype=torch.long)
    gnn_is_heavy_atom_mask[:n_heavy_atoms] = 1

    # 7. Add GNN data in schnetpack format
    properties = {}
    ats = Atoms(numbers=atomic_numbers, positions=atom_positions)
    properties[structure.Z] = torch.tensor(ats.numbers, dtype=torch.long)
    properties[structure.R] = torch.tensor(ats.positions, dtype=torch.float)
    
    # Get neighbors
    nbh_list = atoms_converter._get_neighbors(ats)
    properties[structure.idx_i] = nbh_list.idx_i
    properties[structure.idx_j] = nbh_list.idx_j
    properties[structure.n_atoms] = torch.tensor(n_total_atoms)
    
    # Add the mapping mask to the GNN data
    properties['gnn_is_heavy_atom_mask'] = gnn_is_heavy_atom_mask

    return {
        'atom_mask_tokens': torch.tensor(atom_mask_tokens, dtype=torch.long),
        'gnn_data': properties,
    }

class Uspto50GraphDataModule(Uspto50DataModule):
    """
    Adding 3D graph data (gnn_data_list) and atom-to-token mappings
    (atom_mask_tokens) for use with the GraphBARTModel.
    """

    def __init__(self, gnn_cutoff: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        
        # Build the set of atom token IDs from the tokenizer vocabulary
        self.atom_token_ids = self._build_atom_token_set()
        if not self.atom_token_ids:
            print("Warning: Could not identify any atom tokens in the tokenizer vocabulary.")

        self.atoms_converter = AtomsConverter(device="cpu", cutoff=gnn_cutoff)

    def _build_atom_token_set(self) -> Set[int]:
        """
        Creates a set of token indices that correspond to atoms
        by directly accessing the ChemformerTokenizer's property.
        """

        syntax_tokens_str = {
            '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '=', '/', '\\', '.', '#', '%10', '%11', '%12'
        }

        syntax_token_ids = set()
        for t in syntax_tokens_str:
            syntax_token_ids.add(self.tokenizer.vocabulary[t])

        chem_token_ids = set(self.tokenizer.chem_token_idxs)
        atom_token_ids = chem_token_ids - syntax_token_ids

        return atom_token_ids
    
    def _collate(self, batch_raw: List[Dict[str, Any]], train: bool = True) -> Dict[str, Any]:
        """
        Overrides the base collate function to add hybrid model inputs.
        
        1. Calls super()._collate() to get the fully-formed token batch.
        2. Gets the raw encoder SMILES strings.
        3. Iterates over the batch to run `process_hybrid`.
        4. Adds `gnn_data_list` and `atom_mask_tokens` to the final batch dict.
        """
        
        # --- 1. Get Standard Token Batch ---
        # This calls _AbsDataModule._collate, which calls
        # ReactionListDataModule._transform_batch, which uses
        # Uspto50DataModule._get_sequences.
        # The resulting batch_dict is complete for the transformer.
        batch_dict = super()._collate(batch_raw, train=train)
        
        # --- 2. Get Raw Encoder SMILES ---
        # We need to re-run _get_sequences to get the raw SMILES for
        # RDKit processing. This is fast as it's just list lookups.
        reactants, products = self._get_sequences(batch_raw, train)
        encoder_seqs = products if self.reverse else reactants

        # --- 3. Generate Hybrid GNN Data ---
        gnn_data_list = []
        atom_mask_tokens_list = []
        
        encoder_ids = batch_dict["encoder_input"]
        encoder_pad_mask = batch_dict["encoder_pad_mask"]
        
        seq_len, batch_size = encoder_ids.shape
        
        for i in range(batch_size):
            smiles_string = encoder_seqs[i]
            
            # Get the *unpadded* token IDs for this single item
            # We use the *pad mask* (where True is PAD)
            item_pad_mask = encoder_pad_mask[:, i] # (seq_len,)
            item_token_ids = encoder_ids[:, i]   # (seq_len,)
            
            unpadded_len = (~item_pad_mask).sum().item()
            unpadded_token_ids = item_token_ids[:unpadded_len]

            hybrid_data: Optional[Dict[str, Any]] = None
            if unpadded_len > 0 and smiles_string:
                try:
                    hybrid_data = process_hybrid(
                        unpadded_token_ids, 
                        smiles_string, 
                        self.atom_token_ids,
                        self.atoms_converter
                    )
                except Exception as e:
                    print(f"Error processing SMILES {smiles_string}: {e}")
                    # hybrid_data remains None
            
            if hybrid_data is not None:
                # Success
                gnn_data_list.append(hybrid_data["gnn_data"])
                
                # Pad the atom_mask_tokens back to seq_len
                atom_mask = hybrid_data["atom_mask_tokens"] # (unpadded_len,)
                padded_atom_mask = F.pad(atom_mask, (0, seq_len - unpadded_len), "constant", 0)
                atom_mask_tokens_list.append(padded_atom_mask)
            else:
                # process_hybrid failed
                gnn_data_list.append(None) # Model must be able to handle None
                atom_mask_tokens_list.append(torch.zeros(seq_len, dtype=torch.long))

        # --- 4. Add New Data to Batch ---
        batch_dict["gnn_data_list"] = gnn_data_list
        
        if atom_mask_tokens_list:
            batch_dict["atom_mask_tokens"] = torch.stack(atom_mask_tokens_list, dim=1)
        else:
            # Handle empty batch case
            batch_dict["atom_mask_tokens"] = torch.empty(seq_len, 0, dtype=torch.long)
            
        return batch_dict

# vim: ts=4 sw=4 expandtab

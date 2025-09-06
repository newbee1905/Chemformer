#!/usr/bin/env python3
"""
Comprehensive fine-tuning data preparation script for Chemformer.
Supports USPTO, custom reaction data, and molecular optimization datasets.
"""

import pandas as pd
import numpy as np
import pickle
import os
from rdkit import Chem
from typing import List, Dict, Any, Optional
import argparse

class FineTuneDataPreparer:
    """Prepare data for Chemformer fine-tuning tasks."""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _create_splits(self, df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
        """Create train/val/test splits with specified ratios."""
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        n_total = len(df)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        n_test = n_total - n_train - n_val  # Remaining goes to test
        
        # Shuffle indices
        indices = np.random.permutation(n_total)
        
        # Create splits
        splits = ['train'] * n_total
        for idx in indices[n_train:n_train + n_val]:
            splits[idx] = 'val'
        for idx in indices[n_train + n_val:]:
            splits[idx] = 'test'
        
        df['set'] = splits
    
    def prepare_uspto_data(self, input_file: str, output_name: str, n_samples: Optional[int] = None, 
                          train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> str:
        """
        Prepare USPTO reaction data for fine-tuning.
        
        Args:
            input_file: Path to USPTO pickle file
            output_name: Name for output file (without extension)
            n_samples: Number of samples to extract (None for all)
            train_ratio: Ratio for training set (default: 0.8)
            val_ratio: Ratio for validation set (default: 0.1)
            test_ratio: Ratio for test set (default: 0.1)
        
        Returns:
            Path to prepared pickle file
        """
        print(f"Preparing USPTO data from {input_file}...")
        
        # Load original data
        df = pd.read_pickle(input_file)
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Sample if requested
        if n_samples and n_samples < len(df):
            # Preserve split ratios
            if 'set' in df.columns:
                df = df.groupby('set').apply(
                    lambda x: x.sample(n=min(len(x), n_samples//3), random_state=42)
                ).reset_index(drop=True)
            else:
                df = df.sample(n=n_samples, random_state=42)
            print(f"Sampled to {len(df)} samples")
        
        # Create or update splits with custom ratios
        if 'set' not in df.columns or train_ratio != 0.8 or val_ratio != 0.1 or test_ratio != 0.1:
            print(f"Creating splits with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
            self._create_splits(df, train_ratio, val_ratio, test_ratio)
        else:
            # Fix split names if needed
            if 'set' in df.columns:
                df['set'] = df['set'].replace('valid', 'val')
        
        print(f"Split distribution: {df['set'].value_counts()}")
        
        # Save prepared data
        output_path = os.path.join(self.output_dir, f"{output_name}.pickle")
        df.to_pickle(output_path)
        
        print(f"✅ USPTO data saved to: {output_path}")
        return output_path
    
    def prepare_custom_reaction_data(self, 
                                   reactants: List[str], 
                                   products: List[str], 
                                   reaction_types: Optional[List[str]] = None,
                                   output_name: str = "custom_reactions",
                                   train_ratio: float = 0.8, 
                                   val_ratio: float = 0.1, 
                                   test_ratio: float = 0.1) -> str:
        """
        Prepare custom reaction data for fine-tuning.
        
        Args:
            reactants: List of reactant SMILES
            products: List of product SMILES
            reaction_types: Optional list of reaction type tokens
            output_name: Name for output file
        
        Returns:
            Path to prepared pickle file
        """
        print(f"Preparing custom reaction data with {len(reactants)} reactions...")
        
        # Validate inputs
        if len(reactants) != len(products):
            raise ValueError("Reactants and products must have same length")
        
        # Convert to molecules
        reactants_mol = []
        products_mol = []
        valid_indices = []
        
        for i, (reactant, product) in enumerate(zip(reactants, products)):
            try:
                reactant_mol = Chem.MolFromSmiles(reactant)
                product_mol = Chem.MolFromSmiles(product)
                if reactant_mol is not None and product_mol is not None:
                    reactants_mol.append(reactant_mol)
                    products_mol.append(product_mol)
                    valid_indices.append(i)
            except:
                continue
        
        print(f"Valid reactions: {len(reactants_mol)}")
        
        # Create DataFrame
        data = {
            'reactants_mol': reactants_mol,
            'products_mol': products_mol,
        }
        
        if reaction_types:
            valid_types = [reaction_types[i] for i in valid_indices]
            data['reaction_type'] = valid_types
        
        df = pd.DataFrame(data)
        
        # Create train/val/test splits with custom ratios
        self._create_splits(df, train_ratio, val_ratio, test_ratio)
        
        # Save
        output_path = os.path.join(self.output_dir, f"{output_name}.pickle")
        df.to_pickle(output_path)
        
        print(f"✅ Custom reaction data saved to: {output_path}")
        print(f"Split distribution: {df['set'].value_counts()}")
        return output_path
    
    def prepare_molecular_optimization_data(self,
                                          input_molecules: List[str],
                                          output_molecules: List[str],
                                          property_tokens: List[str],
                                          output_name: str = "mol_opt",
                                          train_ratio: float = 0.8, 
                                          val_ratio: float = 0.1, 
                                          test_ratio: float = 0.1) -> str:
        """
        Prepare molecular optimization data for fine-tuning.
        
        Args:
            input_molecules: List of input molecule SMILES
            output_molecules: List of output molecule SMILES
            property_tokens: List of property tokens (e.g., "high_logp", "low_mw")
            output_name: Name for output file
        
        Returns:
            Path to prepared pickle file
        """
        print(f"Preparing molecular optimization data with {len(input_molecules)} pairs...")
        
        # Validate and convert
        input_mols = []
        output_mols = []
        valid_indices = []
        
        for i, (inp, out) in enumerate(zip(input_molecules, output_molecules)):
            try:
                inp_mol = Chem.MolFromSmiles(inp)
                out_mol = Chem.MolFromSmiles(out)
                if inp_mol is not None and out_mol is not None:
                    input_mols.append(inp_mol)
                    output_mols.append(out_mol)
                    valid_indices.append(i)
            except:
                continue
        
        valid_props = [property_tokens[i] for i in valid_indices]
        
        # Create DataFrame
        df = pd.DataFrame({
            'input_mols': input_mols,
            'output_mols': output_mols,
            'property_tokens': valid_props
        })
        
        # Create splits with custom ratios
        self._create_splits(df, train_ratio, val_ratio, test_ratio)
        
        # Save
        output_path = os.path.join(self.output_dir, f"{output_name}.pickle")
        df.to_pickle(output_path)
        
        print(f"✅ Molecular optimization data saved to: {output_path}")
        print(f"Split distribution: {df['set'].value_counts()}")
        return output_path
    
    def validate_dataset(self, pickle_path: str) -> Dict[str, Any]:
        """
        Validate a prepared dataset.
        
        Args:
            pickle_path: Path to pickle file
        
        Returns:
            Validation results
        """
        print(f"Validating dataset: {pickle_path}")
        
        try:
            df = pd.read_pickle(pickle_path)
            results = {
                'shape': df.shape,
                'columns': list(df.columns),
                'valid': True
            }
            
            # Check splits
            if 'set' in df.columns:
                results['splits'] = df['set'].value_counts().to_dict()
                if 'val' not in df['set'].values:
                    results['warning'] = "No validation samples found"
            
            # Check required columns for different tasks
            if 'reactants_mol' in df.columns and 'products_mol' in df.columns:
                results['task_type'] = 'reaction_prediction'
            elif 'input_mols' in df.columns and 'output_mols' in df.columns:
                results['task_type'] = 'molecular_optimization'
            else:
                results['task_type'] = 'unknown'
            
            print(f"✅ Dataset validation passed")
            print(f"   Shape: {results['shape']}")
            print(f"   Task: {results['task_type']}")
            if 'splits' in results:
                print(f"   Splits: {results['splits']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Dataset validation failed: {e}")
            return {'valid': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Prepare data for Chemformer fine-tuning")
    parser.add_argument("--task", choices=["uspto", "custom_reaction", "mol_opt"], required=True,
                       help="Type of fine-tuning task")
    parser.add_argument("--input", required=True, help="Input file or data")
    parser.add_argument("--output", required=True, help="Output file name")
    parser.add_argument("--n_samples", type=int, help="Number of samples to extract")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)")
    parser.add_argument("--validate", action="store_true", help="Validate the prepared dataset")
    
    args = parser.parse_args()
    
    preparer = FineTuneDataPreparer()
    
    if args.task == "uspto":
        output_path = preparer.prepare_uspto_data(
            args.input, args.output, args.n_samples, 
            args.train_ratio, args.val_ratio, args.test_ratio
        )
    elif args.task == "custom_reaction":
        # For custom reactions, you'd need to provide the data differently
        print("For custom reactions, use the class methods directly in Python")
        return
    elif args.task == "mol_opt":
        print("For molecular optimization, use the class methods directly in Python")
        return
    
    if args.validate:
        preparer.validate_dataset(output_path)

if __name__ == "__main__":
    main()

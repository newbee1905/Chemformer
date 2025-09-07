import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle
import os
import glob

def combine_zinc_data():
    # csv_files = glob.glob("data/zinc/*.csv")
    # dfs = []
    # for file in csv_files:
    #     df = pd.read_csv(file)
    #     dfs.append(df)

    # combined_df = pd.concat(dfs, ignore_index=True)
    # combined_df.to_csv('data/training/zinc_combined.csv', index=False)
    pass

def check_zinc_data():
    csv_files = glob.glob("data/zinc/*.csv")
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        print(df.columns)
        print(df.head())

if __name__ == "__main__":
    combine_zinc_data()
    check_zinc_data()

import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import lmdb
import pandas as pd
from tqdm.auto import tqdm

import gc

def convert_zinc_csvs_to_lmdb(csv_dir: str, lmdb_path: str, map_size: int):
	"""
	Converts a directory of ZINC CSV files into a single LMDB database
	"""

	csv_files = sorted(list(Path(csv_dir).glob("*.csv")))
	if not csv_files:
		print(f"Error: No .csv files found in directory '{csv_dir}'")
		return

	print(f"Found {len(csv_files)} CSV files to process.")

	env = lmdb.open(lmdb_path, map_size=map_size)
	total_records = 0
	split_indices = defaultdict(list)

	with env.begin(write=True) as txn:
		for csv_file in tqdm(csv_files, desc="Processing files"):
			try:
				df = pd.read_csv(csv_file)

				if "smiles" not in df.columns or "set" not in df.columns:
					print(f"Warning: 'smiles' or 'set' column not found in {csv_file}. Skipping.")
					continue

				for _, row in df.iterrows():
					key = str(total_records).encode("utf-8")
					
					record = {"zinc_id": row["zinc_id"], "smiles": row["smiles"]}
					value = pickle.dumps(record)
					
					txn.put(key, value)
					
					split_indices[row["set"]].append(total_records)
					total_records += 1

				del df
				gc.collect()

			except Exception as e:
				print(f"Error processing file {csv_file}: {e}")

		txn.put(b"__len__", str(total_records).encode("utf-8"))
		
		train_idxs = split_indices.get("train", [])
		val_idxs = split_indices.get("val", [])
		test_idxs = split_indices.get("test", [])

		txn.put(b"train_idxs", pickle.dumps(train_idxs))
		txn.put(b"val_idxs", pickle.dumps(val_idxs))
		txn.put(b"test_idxs", pickle.dumps(test_idxs))

		print(f"\nSplit sizes: Train={len(train_idxs)}, Val={len(val_idxs)}, Test={len(test_idxs)}")


	env.close()
	print("\nConversion complete.")
	print(f"Total records written: {total_records}")
	print(f"LMDB database saved at: {lmdb_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Convert a directory of ZINC CSV files to an LMDB database."
	)
	parser.add_argument(
		"--csv_dir",
		type=str,
		required=True,
		help="Path to the directory containing ZINC .csv files.",
	)
	parser.add_argument(
		"--lmdb_path",
		type=str,
		required=True,
		help="Path to the output LMDB directory to be created.",
	)
	parser.add_argument(
		"--map_size",
		type=int,
		default=1024**4,  # Default to 1 TB
		help="Maximum size for the LMDB database in bytes. Default is 1 TB.",
	)
	args = parser.parse_args()

	convert_zinc_csvs_to_lmdb(
		csv_dir=args.csv_dir,
		lmdb_path=args.lmdb_path,
		map_size=args.map_size,
	)

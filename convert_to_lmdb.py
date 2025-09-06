# convert_to_lmdb.py
import argparse
import pickle
from collections import defaultdict

import lmdb
import pandas as pd
from tqdm.auto import tqdm


def convert_pickle_to_lmdb(pickle_path: str, lmdb_path: str, map_size: int):
	"""
	Converts a pandas DataFrame stored in a pickle file to an LMDB database.
	"""

	print(f"Loading DataFrame from {pickle_path}...")
	try:
		df = pd.read_pickle(pickle_path).reset_index(drop=True)
	except Exception as e:
		print(f"Error reading pickle file: {e}")
		return

	num_rows = len(df)
	print(f"DataFrame loaded with {num_rows} rows.")

	split_indices = defaultdict(list)
	has_split = "set" in df.columns
	if has_split:
		print("Found 'set' column, creating data splits...")
		for idx, row in df.iterrows():
			split_indices[row["set"]].append(idx)

		df = df.drop(columns=["set"])

		train_idxs = split_indices.get("train", [])
		val_idxs = split_indices.get("valid", [])
		test_idxs = split_indices.get("test", [])

		print(f"Split sizes: Train={len(train_idxs)}, Val={len(val_idxs)}, Test={len(test_idxs)}")


	print(f"Opening LMDB environment at {lmdb_path}...")
	env = lmdb.open(lmdb_path, map_size=map_size)

	with env.begin(write=True) as txn:
		len_key = b"__len__"
		len_value = str(num_rows).encode("utf-8")
		txn.put(len_key, len_value)
		print(f"Stored dataset length: {num_rows}")

		if has_split:
			txn.put(b"train_idxs", pickle.dumps(train_idxs))
			txn.put(b"val_idxs", pickle.dumps(val_idxs))
			txn.put(b"test_idxs", pickle.dumps(test_idxs))
			print("Stored train, val, and test indices.")

		print("Writing records to LMDB...")
		for idx, row in tqdm(df.iterrows(), total=num_rows, desc="Converting"):
			key = str(idx).encode("utf-8")
			value = pickle.dumps(row.to_dict())
			txn.put(key, value)
			
	env.close()
	print("\nConversion complete.")
	print(f"LMDB database saved at: {lmdb_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Convert a pandas DataFrame pickle file to an LMDB database."
	)
	parser.add_argument(
		"--pickle_path",
		type=str,
		required=True,
		help="Path to the input pandas pickle file (.pkl).",
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
		default=1024**4,  # Default to 1 TB, a very large value
		help="Maximum size for the LMDB database in bytes. Default is 1 TB.",
	)
	args = parser.parse_args()

	convert_pickle_to_lmdb(
		pickle_path=args.pickle_path,
		lmdb_path=args.lmdb_path,
		map_size=args.map_size,
	)

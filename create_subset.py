import argparse
import pandas as pd

def create_subset(
	input_path: str, output_path: str, fraction: float, random_state: int = 42
):
	"""
	Creates a smaller subset of a dataset stored in a pickle file.
	"""
	print(f"Loading DataFrame from {input_path}...")
	try:
		df = pd.read_pickle(input_path)
	except Exception as e:
		print(f"Error reading pickle file: {e}")
		return

	print(f"Original dataset size: {len(df)} rows.")

	has_split = "set" in df.columns

	if has_split:
		print(f"Found 'set' column. Performing stratified sampling of {fraction:.2%}...")

		sampled_df = (
			df.groupby("set", group_keys=False)
			.apply(lambda x: x.sample(frac=fraction, random_state=random_state))
		)
	else:
		print(f"No 'set' column found. Sampling {fraction:.2%} from the entire dataset...")
		sampled_df = df.sample(frac=fraction, random_state=random_state)

	sampled_df = sampled_df.reset_index(drop=True)

	print(f"New dataset size: {len(sampled_df)} rows.")

	if has_split:
		print("\nNew split sizes:")
		print(sampled_df["set"].value_counts())

	print(f"\nSaving subset to {output_path}...")
	try:
		sampled_df.to_pickle(output_path)
		print(f"Subset creation complete. File saved at: {output_path}")
	except Exception as e:
		print(f"Error saving pickle file: {e}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Create a smaller subset of a pandas DataFrame from a pickle file."
	)
	parser.add_argument(
		"--input_path",
		type=str,
		required=True,
		help="Path to the input pandas pickle file (.pkl).",
	)
	parser.add_argument(
		"--output_path",
		type=str,
		required=True,
		help="Path to save the output subset pickle file (.pkl).",
	)
	parser.add_argument(
		"--fraction",
		type=float,
		default=0.01,
		help="The fraction of the dataset to keep. Default is 0.01 (1%).",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="Random seed for sampling to ensure reproducibility. Default is 0.",
	)
	args = parser.parse_args()

	create_subset(
		input_path=args.input_path,
		output_path=args.output_path,
		fraction=args.fraction,
		random_state=args.seed,
	)

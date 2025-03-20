"""clean_parallel_corpus.py - Clean parallel corpus by removing misaligned abstracts"""
import pandas as pd
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Clean parallel corpus by removing misaligned abstracts")
parser.add_argument("--input", required=True, help="Path to the parallel corpus (TSV format)")
parser.add_argument("--output", required=True, help="Path to save the cleaned corpus")
args = parser.parse_args()

# Load data
df = pd.read_csv(args.input, sep="\t", encoding="utf-8")

# Ensure required columns exist
if "Russian" not in df.columns or "English" not in df.columns:
    raise ValueError("Error: 'Russian' and 'English' columns not found in the input file.")

# Compute word lengths
df["Russian_Length"] = df["Russian"].apply(lambda x: len(x.split()))
df["English_Length"] = df["English"].apply(lambda x: len(x.split()))

# Compute length ratio
df["Length_Ratio"] = df["English_Length"] / df["Russian_Length"]

# Filter criteria
filtered_df = df[(df["Length_Ratio"] >= 0.5) & (df["Length_Ratio"] <= 2.0)]
#filtered_df = filtered_df[(filtered_df["Russian_Length"] >= 10) & (filtered_df["English_Length"] >= 10)]

# Remove duplicates based on Russian text
filtered_df = filtered_df.drop_duplicates(subset=["Russian"], keep="first")

# Save cleaned dataset
filtered_df.to_csv(args.output, sep="\t", index=False)
print(f"Cleaned parallel corpus saved to {args.output}")

"""extract_russian.py"""
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Extract Russian text for SentencePiece training.")
parser.add_argument("--input", required=True, help="Input parallel corpus (TSV format)")
parser.add_argument("--output", required=True, help="Output file with only Russian sentences")
args = parser.parse_args()

# Load parallel corpus
df = pd.read_csv(args.input, sep="\t", encoding="utf-8")

# Ensure "Russian" column exists
if "Russian" not in df.columns:
    raise ValueError("Column 'Russian' not found in input file!")

# Save Russian sentences only
df["Russian"].to_csv(args.output, index=False, header=False, encoding="utf-8")

print(f"Extracted Russian sentences saved to {args.output}")

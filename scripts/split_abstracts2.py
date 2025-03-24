"""split_abstracts2.py - Split parallel abstracts into sentence pairs"""
import pandas as pd
import re
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Split parallel abstracts into sentence pairs")
parser.add_argument("--input", required=True, help="Path to the parallel corpus (TSV format)")
parser.add_argument("--output", required=True, help="Path to save the sentence-aligned corpus")
args = parser.parse_args()

# Load data
df = pd.read_csv(args.input, sep="\t", encoding="utf-8")

# Ensure required columns exist
if "Russian" not in df.columns or "English" not in df.columns:
    raise ValueError("Error: 'Russian' and 'English' columns not found in the input file.")

# Simple sentence splitting function using regex (handles '.', '!', '?')
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]  # Remove empty sentences

# Process each abstract into sentences
sentence_pairs = []
for _, row in df.iterrows():
    ru_sentences = split_sentences(row["Russian"])
    en_sentences = split_sentences(row["English"])
    
    # Ensure the number of sentences match
    min_len = min(len(ru_sentences), len(en_sentences))
    
    for i in range(min_len):
        sentence_pairs.append((ru_sentences[i], en_sentences[i]))

# Create new DataFrame
sentence_df = pd.DataFrame(sentence_pairs, columns=["Russian", "English"])

# Save the new sentence-aligned dataset
sentence_df.to_csv(args.output, sep="\t", index=False)
print(f"Sentence-aligned corpus saved to {args.output}")
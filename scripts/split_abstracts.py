"""split_abstracts.py - Split parallel abstracts into sentence pairs"""

import re
import argparse
import pandas as pd


# Argument parser
parser = argparse.ArgumentParser(
    description="Split parallel abstracts into sentence pairs"
)
parser.add_argument(
    "--input", required=True, help="Path to the parallel corpus (TSV format)"
)
parser.add_argument(
    "--output", required=True, help="Path to save the sentence-aligned corpus"
)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.input, sep="\t", encoding="utf-8")

# Ensure required columns exist
if "Russian" not in df.columns or "English" not in df.columns:
    raise ValueError(
        "Error: 'Russian' and 'English' columns not found in the input file."
    )


def is_fragment(text):
    """Check if the text is a fragment (short sentence or abbreviation)"""
    return len(text.split()) < 4 and re.search(r"\b(им|им\.|[А-Яа-я]\.)\b", text)


# Simple sentence splitting function using regex (handles '.', '!', '?')
def split_sentences(text):
    """Split text into sentences using regex with some additional rules"""
    if not isinstance(text, str):
        return []

    # Known non-breaking abbreviations
    abbreviations = [
        "г",
        "гг",
        "долл",
        "млн",
        "млрд",
        "рис",
        "табл",
        "им",
        "т",
        "и др",
        "т.д",
        "т.п",
        "см",
        "о.е",
        "н.э",
        "т.е",
    ]

    # Protect them: replace "." with "<DOT>" temporarily
    for abbr in abbreviations:
        text = re.sub(rf"\b{abbr}\.", f"{abbr}<DOT>", text, flags=re.IGNORECASE)

    # Split on real sentence boundaries
    text = re.sub(r"(?<=[.!?])\s+(?=[А-ЯA-Z])", "<SPLIT>", text)

    # Restore abbreviation periods
    text = text.replace("<DOT>", ".")

    # Final split
    sentences = text.split("<SPLIT>")
    return [s.strip() for s in sentences if s.strip()]


# Process each abstract into sentences
sentence_pairs = []
for idx, row in df.iterrows():
    ru_sentences = split_sentences(row["Russian"])
    en_sentences = split_sentences(row["English"])

    min_len = min(len(ru_sentences), len(en_sentences))
    for i in range(min_len):
        sentence_pairs.append((idx, ru_sentences[i], en_sentences[i]))

# Create new DataFrame with index
sentence_df = pd.DataFrame(sentence_pairs, columns=["AbstractID", "Russian", "English"])

# Remove fragments (short sentences or abbreviations)
sentence_df = df[~df["Russian"].apply(is_fragment)]

# Save the new sentence-aligned dataset
sentence_df.to_csv(args.output, sep="\t", index=False)
print(f"Sentence-aligned corpus saved to {args.output}")

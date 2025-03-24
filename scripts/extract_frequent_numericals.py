"""extract_frequent_numericals.py - Extract frequent numericals from a parallel corpus"""
import pandas as pd
import re
from collections import Counter

def extract_numbers(text, min_digits=2):
    """Extract digit-only tokens of a certain minimum length"""
    return re.findall(rf"\b\d{{{min_digits},}}\b", str(text))

def main(input_path, output_path="frequent_numericals.txt", min_digits=2, min_freq=3):
    df = pd.read_csv(input_path, sep="\t", encoding="utf-8")

    # Extract numericals from the Russian column
    all_numbers = []
    for text in df["Russian"]:
        all_numbers.extend(extract_numbers(text, min_digits))

    # Count frequencies
    freq = Counter(all_numbers)

    # Filter by frequency
    filtered = [(n, c) for n, c in freq.items() if c >= min_freq]
    filtered.sort(key=lambda x: -x[1])  # Sort by frequency descending

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        for number, count in filtered:
            f.write(f"{number}\n")

    print(f"✅ Saved {len(filtered)} frequent numericals to: {output_path}")
    print("📌 Examples:")
    for number, count in filtered[:10]:
        print(f"{number}\t{count}")

if __name__ == "__main__":
    # Customize thresholds here
    main(
        input_path="data/parallel_corpus.tsv",
        output_path="user_defined_numericals.txt",
        min_digits=2,
        min_freq=3
    )

"""abstract_analysis.py - Analyze parallel corpus of English and Russian abstracts"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Argument parser
parser = argparse.ArgumentParser(description="Analyze parallel corpus")
parser.add_argument(
    "--input", required=True, help="Path to the parallel corpus (TSV format)"
)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.input, sep="\t", encoding="utf-8")

# Ensure required columns exist
if "Russian" not in df.columns or "English" not in df.columns:
    raise ValueError(
        "Error: 'Russian' and 'English' columns not found in the input file."
    )

# Compute raw text length
df["Russian_Length"] = df["Russian"].apply(lambda x: len(x.split()))
df["English_Length"] = df["English"].apply(lambda x: len(x.split()))

# Compute length ratio
df["Length_Ratio"] = df["English_Length"] / df["Russian_Length"]

# Display basic statistics
print("\n===== Dataset Statistics =====")
print(df.describe())

# Identify potential misalignments (large length differences)
outliers = df[(df["Length_Ratio"] > 2.0) | (df["Length_Ratio"] < 0.5)]
print(f"\nNumber of misaligned abstracts: {len(outliers)}")

# Plot distribution of text lengths
sns.histplot(df["Russian_Length"], bins=30, kde=True, label="Russian", color="blue")
sns.histplot(
    df["English_Length"], bins=30, kde=True, label="English", color="orange", alpha=0.6
)
plt.legend()
plt.title("Distribution of Abstract Lengths (Words)")
plt.xlabel("Number of Words")
plt.ylabel("Count")
plt.show()

# Plot length ratio
sns.histplot(df["Length_Ratio"], bins=30, kde=True)
plt.axvline(x=1.0, color="r", linestyle="--")  # Perfect alignment
plt.axvline(x=2.0, color="g", linestyle="--")  # 2x length difference
plt.axvline(x=0.5, color="g", linestyle="--")  # 0.5x length difference
plt.title("Distribution of English-to-Russian Length Ratios")
plt.xlabel("Length Ratio (English / Russian)")
plt.ylabel("Count")
plt.show()

# Save outliers for inspection
outliers.to_csv("misaligned_abstracts.tsv", sep="\t", index=False)
print("Misaligned abstracts saved to 'misaligned_abstracts.tsv'")

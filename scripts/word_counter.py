"""word_counter.py - a simple script to count word frequencies in a text file."""

from collections import Counter
import re

# Load the translation output file
FILE_PATH = "data/wmt_test_translations_with_beam.txt"
# file_path = "data/wmt_test_translations_with_beam_fixed.txt"
# Read and preprocess text
with open(FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read().lower()  # Convert to lowercase for consistency

# Tokenize (simple split by whitespace & remove punctuation)
words = re.findall(r"\b\w+\b", text)

# Count word frequencies
word_counts = Counter(words)

# Display the top 50 most frequent words
print("Top 50 most common words in translations:")
for word, freq in word_counts.most_common(50):
    print(f"{word}: {freq}")

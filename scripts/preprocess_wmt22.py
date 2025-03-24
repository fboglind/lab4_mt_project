"""preprocess_wmt22.py - Preprocess WMT22 data for training and testing"""

import os
import glob

# Paths
DATA_DIR = "abstracts/"  # Adjust this if necessary
OUTPUT_PARALLEL_FILE = "data/parallel_corpus.tsv"  # Parallel training corpus
OUTPUT_TEST_RAW_FILE = "data/test_raw_ru.txt"  # Raw Russian test data

# Collect all file names
ru_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_ru.txt")))
en_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_en.txt")))


# Ensure we have matching pairs
assert len(en_files) == len(ru_files), "Mismatch between English and Russian files!"

# Create a dictionary to store matched pairs
paired_data = {}

# Process Russian files
for ru_file in ru_files:
    pmid = os.path.basename(ru_file).split("_")[0]  # Extract PMID
    with open(ru_file, "r", encoding="utf-8") as f:
        ru_text = f.read().strip()
        if pmid not in paired_data:
            paired_data[pmid] = {}  # Initialize dictionary for this PMID
        paired_data[pmid]["ru"] = ru_text  # Add Russian text

# Process English files
for en_file in en_files:
    pmid = os.path.basename(en_file).split("_")[0]  # Extract PMID
    with open(en_file, "r", encoding="utf-8") as f:
        en_text = f.read().strip()
        if pmid not in paired_data:
            paired_data[pmid] = {}  # Initialize dictionary for this PMID
        paired_data[pmid]["en"] = en_text  # Add English text

# Write parallel corpus to file
with open(OUTPUT_PARALLEL_FILE, "w", encoding="utf-8") as out_f:
    out_f.write("Russian\tEnglish\n")  # Header
    for pmid, texts in paired_data.items():
        if "en" in texts and "ru" in texts:  # Ensure both languages are present
            out_f.write(f"{texts['ru']}\t{texts['en']}\n")

print(
    f"Preprocessing complete. Saved {len(paired_data)} sentence pairs to {OUTPUT_PARALLEL_FILE}"
)

# Extract test set (raw Russian abstracts)
TEST_SIZE = 50  # Adjust this as needed
test_ru_files = ru_files[-TEST_SIZE:]  # Assume last 'test_size' files are test data

with open(OUTPUT_TEST_RAW_FILE, "w", encoding="utf-8") as test_f:
    for ru_file in test_ru_files:
        with open(ru_file, "r", encoding="utf-8") as ru_f:
            test_f.write(ru_f.read().strip() + "\n")

print(f"Raw test Russian abstracts saved to {OUTPUT_TEST_RAW_FILE}")

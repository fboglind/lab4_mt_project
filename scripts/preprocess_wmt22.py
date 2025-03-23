import os
import glob

# Paths
data_dir = "abstracts/"  # Adjust this if necessary
output_parallel_file = "data/parallel_corpus.tsv"  # Parallel training corpus
output_test_raw_file = "data/test_raw_ru.txt"  # Raw Russian test data

# Collect all file names
ru_files = sorted(glob.glob(os.path.join(data_dir, "*_ru.txt")))
en_files = sorted(glob.glob(os.path.join(data_dir, "*_en.txt")))


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
with open(output_parallel_file, "w", encoding="utf-8") as out_f:
    out_f.write("Russian\tEnglish\n")  # Header
    for pmid, texts in paired_data.items():
        if "en" in texts and "ru" in texts:  # Ensure both languages are present
            out_f.write(f"{texts['ru']}\t{texts['en']}\n")

print(f"Preprocessing complete. Saved {len(paired_data)} sentence pairs to {output_parallel_file}")

# Extract test set (raw Russian abstracts)
test_size = 50  # Adjust this as needed
test_ru_files = ru_files[-test_size:]  # Assume last 'test_size' files are test data

with open(output_test_raw_file, "w", encoding="utf-8") as test_f:
    for ru_file in test_ru_files:
        with open(ru_file, "r", encoding="utf-8") as ru_f:
            test_f.write(ru_f.read().strip() + "\n")

print(f"Raw test Russian abstracts saved to {output_test_raw_file}")

import os
import glob

# Paths
data_dir = "abstracts/"  # Adjust this if necessary
output_file = "parallel_corpus.tsv"  # Tab-separated output file

# Collect all file names
en_files = glob.glob(os.path.join(data_dir, "*_en.txt"))
ru_files = glob.glob(os.path.join(data_dir, "*_ru.txt"))

# Create a dictionary to store matched pairs
paired_data = {}

# Process English files
for en_file in en_files:
    pmid = os.path.basename(en_file).split("_")[0]  # Extract PMID
    with open(en_file, "r", encoding="utf-8") as f:
        en_text = f.read().strip()
        paired_data[pmid] = {"en": en_text}

# Process Russian files
for ru_file in ru_files:
    pmid = os.path.basename(ru_file).split("_")[0]  # Extract PMID
    with open(ru_file, "r", encoding="utf-8") as f:
        ru_text = f.read().strip()
        if pmid in paired_data:
            paired_data[pmid]["ru"] = ru_text  # Add Russian text

# Write to a tab-separated file (only fully matched pairs)
with open(output_file, "w", encoding="utf-8") as out_f:
    out_f.write("English\tRussian\n")  # Header
    for pmid, texts in paired_data.items():
        if "en" in texts and "ru" in texts:
            out_f.write(f"{texts['en']}\t{texts['ru']}\n")

print(f"Preprocessing complete. Saved {len(paired_data)} sentence pairs to {output_file}")


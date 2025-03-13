import os
import pandas as pd
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

# Paths
input_file = "cleaned_parallel_corpus.tsv"
bpe_model_file = "bpe_model.codes"
output_file = "bpe_parallel_corpus.tsv"
num_operations = 32000  # Adjust based on dataset size

# Read the dataset
df = pd.read_csv(input_file, sep="\t", encoding="utf-8")

# Merge English & Russian text into one corpus for training BPE
merged_text = df["English"].tolist() + df["Russian"].tolist()

# Train BPE model
with open("bpe_training_data.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(merged_text))

with open("bpe_training_data.txt", "r", encoding="utf-8") as infile, open(bpe_model_file, "w", encoding="utf-8") as outfile:
    learn_bpe(infile, outfile, num_operations)

# Load trained BPE model
with open(bpe_model_file, "r", encoding="utf-8") as bpe_file:
    bpe = BPE(bpe_file)

# Apply BPE to dataset
df["English"] = df["English"].apply(lambda x: " ".join(bpe.process_line(x)))
df["Russian"] = df["Russian"].apply(lambda x: " ".join(bpe.process_line(x)))

# Save BPE-encoded data
df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")

print(f"BPE training and encoding complete! Saved to {output_file}")


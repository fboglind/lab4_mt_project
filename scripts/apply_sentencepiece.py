import sentencepiece as spm
import pandas as pd

# Paths
train_file = "parallel_corpus.tsv"  # Original training data
output_file = "spm_parallel_corpus.tsv"  # Tokenized output file
sp_model = "spm_ru_en.model"

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file=sp_model)

# Load dataset
df = pd.read_csv(train_file, sep="\t", encoding="utf-8")

# Apply SentencePiece tokenization
df["Russian"] = df["Russian"].apply(lambda x: " ".join(sp.encode(x, out_type=str)))
df["English"] = df["English"].apply(lambda x: " ".join(sp.encode(x, out_type=str)))

# Save tokenized data
df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")

print(f"SentencePiece tokenization complete. Saved to {output_file}")


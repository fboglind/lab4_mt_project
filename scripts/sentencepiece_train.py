import sentencepiece as spm
import os

def load_user_defined_symbols(tsv_path, max_terms=1000):
    with open(tsv_path, encoding="utf-8") as f:
        return [line.strip().split("\t")[0] for line in f.readlines()[:max_terms]]

# Paths
train_file = "data/sentence_aligned_corpus.tsv"
sp_model_prefix = "spm_ru_en_joint"
vocab_size = 30000  # Adjust if needed

output_dir = "models/"

# Load user-defined symbols from file
user_defined_symbols = load_user_defined_symbols("latin_terms.tsv")
print(f"Loaded {len(user_defined_symbols)} user-defined symbols")
# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input=train_file,
    model_prefix=os.path.join(output_dir, "spm_ru_only"),
    vocab_size=vocab_size,
    character_coverage=0.9995,  # Adjust for coverage of Russian/English text
    model_type="unigram",  # Options: bpe, unigram, char, word
    input_sentence_size=0, #Use all available data
    shuffle_input_sentence=True
    user_defined_symbols=user_defined_symbols
)

print(f"SentencePiece model trained: {sp_model_prefix}.model")


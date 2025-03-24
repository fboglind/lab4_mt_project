"""sentencepiece_train.py - Trains a SentencePiece model on a parallel corpus"""
import os
import sentencepiece as spm


def load_user_defined_symbols(tsv_path, max_symbols=5000):
    """Load user-defined symbols from file
    Args:
        tsv_path (str): Path to TSV file with symbols
        max_symbols (int): Maximum number of symbols to load
    Returns:
        list[str]: List of symbols"""
    with open(tsv_path, encoding="utf-8") as f:
        return [line.strip().split("\t")[0] for line in f if line.strip()][:max_symbols]


# Load user-defined symbols from file
user_defined_symbols = load_user_defined_symbols("data/user_defined_terms.tsv")
print(f"Loaded {len(user_defined_symbols)} user-defined symbols")
# Paths
TRAIN_FILE = "data/sentence_aligned_corpus.tsv"
SP_MODEL_PREFIX = "spm_ru_en"
vocab_size = len(user_defined_symbols)+30000
OUTPUT_DIR = "models/"

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input=TRAIN_FILE,
    model_prefix=os.path.join(OUTPUT_DIR, "spm_ru_en"),
    vocab_size=vocab_size,
    character_coverage=0.9995,  # Adjust for coverage of Russian/English text
    model_type="unigram",  # Options: bpe, unigram, char, word
    input_sentence_size=0, #Use all available data
    shuffle_input_sentence=True,
    user_defined_symbols=user_defined_symbols
)

print(f"SentencePiece model trained: {SP_MODEL_PREFIX}.model")

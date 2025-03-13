import sentencepiece as spm

# Paths
train_file = "parallel_corpus.tsv"
sp_model_prefix = "spm_ru_en"
vocab_size = 32000  # Adjust if needed

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input=train_file,
    model_prefix=sp_model_prefix,
    vocab_size=vocab_size,
    character_coverage=0.9995,  # Adjust for coverage of Russian/English text
    model_type="unigram",  # Options: bpe, unigram, char, word
    input_sentence_size=1000000,  # Use a subset of sentences if the dataset is large
    shuffle_input_sentence=True
)

print(f"SentencePiece model trained: {sp_model_prefix}.model")


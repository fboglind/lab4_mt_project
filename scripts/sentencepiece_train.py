import sentencepiece as spm
import os
# Paths
train_file = "data/parallel_corpus.tsv"
sp_model_prefix = "spm_ru_only"
vocab_size = 30000  # Adjust if needed

output_dir = "models/"
# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input=train_file,
    model_prefix=os.path.join(output_dir, "spm_ru_only"),
    vocab_size=vocab_size,
    character_coverage=0.9995,  # Adjust for coverage of Russian/English text
    model_type="unigram",  # Options: bpe, unigram, char, word
    #input_sentence_size=1000000,  # Use a subset of sentences if the dataset is large
    input_sentence_size=0, #Use all available data
    shuffle_input_sentence=True
)

print(f"SentencePiece model trained: {sp_model_prefix}.model")


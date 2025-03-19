"""
apply_sentencepiece.py - Tokenize text data using a trained SentencePiece model.

This script applies SentencePiece tokenization to:
1. A **parallel corpus** (Russian-English), where:
   - By default, only the **Russian** column is tokenized.
   - If `--to_parallel` is specified, both Russian and English columns are tokenized.
2. A **test set** (Russian only), producing tokenized input for inference.

Usage Examples:
- Tokenize only Russian in a parallel corpus:
    python3 apply_sentencepiece.py --input data/parallel_corpus.tsv --output data/spm_parallel_corpus.tsv --model models/spm_ru.model

- Tokenize both Russian and English in a parallel corpus:
    python3 apply_sentencepiece.py --input data/parallel_corpus.tsv --output data/spm_parallel_corpus.tsv --model models/spm_ru.model --to_parallel

- Tokenize a test set for inference:
    python3 apply_sentencepiece.py --input data/test_raw_ru.txt --output data/test_preprocessed_ru.txt --model models/spm_ru.model --is_test_set
"""

import sentencepiece as spm
import argparse
import pandas as pd
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Apply SentencePiece tokenization to text data.")
parser.add_argument("--input", required=True, help="Input file (raw text or parallel corpus)")
parser.add_argument("--output", required=True, help="Output file (tokenized text)")
parser.add_argument("--model", default="models/spm_ru.model", help="SentencePiece model file")
parser.add_argument("--to_parallel", action="store_true", help="Specify if output should be a parallel model")
parser.add_argument("--is_test_set", action="store_true", help="Specify if input is a test set")
args = parser.parse_args()

# Check if input file exists
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Error: Input file '{args.input}' not found.")

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file=args.model)

# Process input file
if not args.is_test_set:
    # Parallel corpus case (training data)
    df = pd.read_csv(args.input, sep="\t", encoding="utf-8")

    # Ensure Russian column exists
    if "Russian" not in df.columns:
        raise ValueError("Error: Column 'Russian' not found in input file. Check if input is a valid parallel corpus.")

    # Apply SentencePiece tokenization to Russian
    df["Russian"] = df["Russian"].apply(lambda x: " ".join(sp.encode(x, out_type=str)))

    # Optionally tokenize English
    if args.to_parallel:
        print("Tokenizing PARALLEL columns...")
        if "English" not in df.columns:
            raise ValueError("Error: Column 'English' not found in input file.")
        df["English"] = df["English"].apply(lambda x: " ".join(sp.encode(x, out_type=str)))

    # Save tokenized corpus
    df.to_csv(args.output, sep="\t", index=False, encoding="utf-8")
    print(f"SentencePiece tokenization complete. Processed {len(df)} sentences. Saved to {args.output}")

else:
    # Single-column text file (test data)
    with open(args.input, "r", encoding="utf-8") as f_in, open(args.output, "w", encoding="utf-8") as f_out:
        for line in f_in:
            tokenized = " ".join(sp.encode(line.strip(), out_type=str))
            f_out.write(tokenized + "\n")

    num_lines = sum(1 for _ in open(args.input, "r", encoding="utf-8"))
    print(f"SentencePiece tokenization complete. Processed {num_lines} TEST sentences. Saved to {args.output}")

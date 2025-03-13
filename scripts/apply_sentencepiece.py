import sentencepiece as spm
import argparse
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Apply SentencePiece tokenization to text data.")
parser.add_argument("--input", required=True, help="Input file (raw text or parallel corpus)")
parser.add_argument("--output", required=True, help="Output file (tokenized text)")
parser.add_argument("--model", default="models/spm_ru_en.model", help="SentencePiece model file")
parser.add_argument("--is_parallel", action="store_true", help="Specify if input is a parallel corpus")
args = parser.parse_args()

# Load SentencePiece model
sp = spm.SentencePieceProcessor(model_file=args.model)

# Process input file
if args.is_parallel:
    # Parallel corpus case (training data)
    df = pd.read_csv(args.input, sep="\t", encoding="utf-8")

    # Apply SentencePiece tokenization
    df["Russian"] = df["Russian"].apply(lambda x: " ".join(sp.encode(x, out_type=str)))
    df["English"] = df["English"].apply(lambda x: " ".join(sp.encode(x, out_type=str)))

    # Save tokenized parallel corpus
    df.to_csv(args.output, sep="\t", index=False, encoding="utf-8")

else:
    # Single-column text file (test data)
    with open(args.input, "r", encoding="utf-8") as f_in, open(args.output, "w", encoding="utf-8") as f_out:
        for line in f_in:
            tokenized = " ".join(sp.encode(line.strip(), out_type=str))
            f_out.write(tokenized + "\n")

print(f"SentencePiece tokenization complete. Saved to {args.output}")

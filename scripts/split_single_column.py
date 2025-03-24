"""split single column.py"""
import re
import argparse


def split_sentences(text):
    """Split text into sentences using regex, handling abbreviations"""
    if not isinstance(text, str):
        return []

    abbreviations = [
        "e.g", "i.e", "etc", "Mr", "Ms", "Dr", "vs", "Fig", "Eq", "Ref", "No", "et al", "Inc", "Ltd"
    ]

    for abbr in abbreviations:
        text = re.sub(rf"\b{abbr}\.", f"{abbr}<DOT>", text, flags=re.IGNORECASE)

    text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "<SPLIT>", text)

    text = text.replace("<DOT>", ".")
    sentences = text.split("<SPLIT>")
    return [s.strip() for s in sentences if s.strip()]


def split_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        paragraphs = [line.strip() for line in f if line.strip()]

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for para in paragraphs:
            for sent in split_sentences(para):
                f_out.write(sent + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a single-language text file into sentences")
    parser.add_argument("--input", required=True, help="Path to raw input file")
    parser.add_argument("--output", required=True, help="Path to output sentence-per-line file")
    args = parser.parse_args()

    split_file(args.input, args.output)

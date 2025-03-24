"""extract_numbers_and_latin.py"""
import pandas as pd
import re
from collections import Counter


def extract_latin_words(text):
    # Match Latin script words (≥ 2 characters to skip things like "A", "B")
    return re.findall(r'\b[A-Za-z][A-Za-z0-9\-]{1,}\b', text)

def extract_numbers(text, min_digits=2):
    """Extract digit-only tokens of a certain minimum length"""
    return re.findall(rf"\b\d{{{min_digits},}}\b", str(text))

def filter_numbers_by_frequency(numbers, min_freq=3):
    """Filter numbers by their frequency"""
    counter = Counter(numbers)
    return [num for num, freq in counter.items() if freq >= min_freq]

def score_shared_terms(ru, en):
    ru_terms = set(extract_latin_words(ru))
    en_terms = set(extract_latin_words(en))
    return len(ru_terms & en_terms)

def build_shared_terms_df(df):
    """Builds a DataFrame of shared terms between Russian and English sentences"""
    pairs = set()
    for i, row in df.iterrows():
        ru_terms = extract_latin_words(row["Russian"])
        en_terms = extract_latin_words(row["English"])
        for rt in ru_terms:
            if rt in en_terms:
                pairs.add((rt, rt))  # Direct transfer
                

    return pd.DataFrame(list(pairs), columns=["Russian", "English"])


def main(input_path, output_path=None, min_freq=2):
    df = pd.read_csv(input_path, sep="\t", encoding="utf-8")
    counter = Counter()

    for text in df["Russian"]:
        words = extract_latin_words(str(text))
        counter.update(words)

    # Sort and filter
    terms = [(term, freq) for term, freq in counter.items() if freq >= min_freq]
    terms.sort(key=lambda x: x[1], reverse=True)

    # Display top results
    print("Top Latin-script terms:")
    for term, freq in terms[:50]:
        print(f"{term:<20} {freq}")

    # Optional: save to file
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for term, freq in terms:
                f.write(f"{term}\t{freq}\n")
        print(f"\nSaved {len(terms)} terms to {output_path}")
    
    # Optional: build shared terms DataFrame
    shared_terms_df = build_shared_terms_df(df)
    shared_terms_df.to_csv("data/shared_latin_terms.tsv", sep="\t", index=False, encoding="utf-8")
    print(f"Shared terms saved to: data/shared_latin_terms.tsv")

    
if __name__ == "__main__":
    main("data/sentence_aligned_corpus.tsv", output_path="data/latin_terms.tsv")



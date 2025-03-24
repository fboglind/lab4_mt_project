"""extract_numbers_and_latin.py - Extract Latin-script terms
and frequent numbers from Russian text and saves to file"""

import re
from collections import Counter
import pandas as pd


def extract_latin_words(text):
    """Extract Latin-script words from a string
    Args:
        text (str): The input text

    Returns:
        list: A list of Latin-script words
    """
    # Match Latin script words (≥ 2 characters to skip things like "A", "B")
    return re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{1,}\b", text)


def extract_numbers(text, min_digits=3):
    """Extract digit-only tokens of a certain minimum length
    Args:
        text (str): The input text
        min_digits (int): The minimum number of digits in a token

    Returns:
        list: A list of digit-only tokens
    """
    return re.findall(rf"\b\d{{{min_digits},}}\b", str(text))


def filter_numbers_by_frequency(numbers, min_freq=3):
    """Filter numbers by their frequency

    Args:
        numbers (list): A list of (number, frequency) tuples
        min_freq (int): The minimum frequency to keep

        Returns:
        list: A list of numbers that meet the minimum frequency requirement"""
    counter = Counter(numbers)
    return [num for num, freq in counter.items() if freq >= min_freq]


def score_shared_terms(ru, en):
    """Scores the number of shared terms between two strings

    Args:
        ru (str): A Russian string
        en (str): An English string

    Returns:
        int: The number of shared terms
    """
    ru_terms = set(extract_latin_words(ru))
    en_terms = set(extract_latin_words(en))
    return len(ru_terms & en_terms)


def build_shared_terms_df(df):
    """Builds a DataFrame of shared terms between Russian and English sentences

    Args:
        df (DataFrame): A DataFrame with "Russian" and "English" columns"

    Returns:
        DataFrame: A DataFrame with shared terms

    """
    pairs = set()
    for _, row in df.iterrows():
        ru_terms = extract_latin_words(row["Russian"])
        en_terms = extract_latin_words(row["English"])
        for rt in ru_terms:
            if rt in en_terms:
                pairs.add((rt, rt))  # Direct transfer
            # if rt == et:
            #     pairs.add((rt, et))  # Add matching terms instead

    return pd.DataFrame(list(pairs), columns=["Russian", "English"])


def main(input_path, output_path=None, min_freq=2):
    """Extract Latin-script terms and frequent numbers from Russian text and saves to file"""
    df = pd.read_csv(input_path, sep="\t", encoding="utf-8")
    word_counter = Counter()
    number_counter = Counter()

    for text in df["Russian"]:
        words = extract_latin_words(str(text))
        numbers = extract_numbers(str(text))
        word_counter.update(words)
        number_counter.update(numbers)

    # Sort and filter - using different methods here
    terms = [(term, freq) for term, freq in word_counter.items() if freq >= min_freq]
    terms.sort(key=lambda x: x[1], reverse=True)

    numbers = [
        (number, freq) for number, freq in number_counter.items() if freq >= min_freq
    ]
    numbers.sort(key=lambda x: x[1], reverse=True)
    # numbers = filter_numbers_by_frequency(numbers)

    # Display top terms
    print("Top Latin-script terms:")
    for term, freq in terms[:50]:
        print(f"{term:<20} {freq}")

    # Display top numbers
    print("\nTop numbers:")
    for number, freq in numbers[:10]:  # Unpack the tuple
        print(f"{number:<20} {freq}")  # Format like terms

    # Save to file
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            # Write terms
            for term, freq in terms:
                f.write(f"{term}\t{freq}\n")

            # Write numbers
            f.write("\nNumbers:\n")
            for (
                number,
                freq,
            ) in numbers:  # Ensure numbers are written in the same format
                f.write(f"{number}\t{freq}\n")

        print(f"\nSaved {len(terms)} terms and {len(numbers)} numbers to {output_path}")

    # # Optional: build shared terms DataFrame
    # shared_terms_df = build_shared_terms_df(df)
    # shared_terms_df.to_csv("data/shared_latin_terms.tsv", sep="\t", index=False, encoding="utf-8")
    # print(f"Shared terms saved to: data/shared_latin_terms.tsv")


if __name__ == "__main__":
    main("data/sentence_aligned_corpus.tsv", output_path="data/user_defined_terms.tsv")

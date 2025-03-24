"""clean_russian_openers.py"""

import re
import pandas as pd


# More specific patterns come first
openers_to_remove = [
    r"^Резюме\s+Цель\s+исследования\.?\s*[^А-Яа-яA-Za-z]*\s*",  # Most specific
    r"^Резюме\s+Цель\s*[^А-Яа-яA-Za-z]*\s*",
    r"^Резюме\s+Проблема\s*[^А-Яа-яA-Za-z]*\s*",
    r"^Цель\s+исследования\s*[^А-Яа-яA-Za-z]*\s*",
    r"^Цель\s*-\s*",
    r"^Цель\.\s*",
    r"^Аннотация\s*[^А-Яа-яA-Za-z]*\s*",
    r"^Введение\s*[^А-Яа-яA-Za-z]*\s*",
    r"^Резюме\s*[^А-Яа-яA-Za-z]*\s*",  # Most general
]

# Section headers to remove, even if they occur mid-text
section_headers = [
    r"\bМатериалы и методы\b[\.:]?",
    r"\bМатериал и методы\b[\.:]?",
    r"\bРезультаты и обсуждение\b[\.:]?",
    r"\bРезультаты и заключение\b[\.:]?",
    r"\bРезультаты\b[\.:]?",
    r"\bВыводы\b[\.:]?",
    r"\bВывод\b[\.:]?",
    r"\bЗаключение\b[\.:]?",
    r"\bЦель исследования\b[\.:]?",
]


def remove_section_headers(text):
    """Remove section headers from Russian text"""
    for pattern in section_headers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def clean_russian_line(text):
    """Remove openers and headers from Russian text"""
    if not isinstance(text, str):
        return text

    # First, remove openers at the beginning
    for pattern in openers_to_remove:
        new_text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        if new_text != text:
            text = new_text
            break

    # Then remove any standalone or mid-text section headers
    text = remove_section_headers(text)

    return text.strip()


def clean_parallel_corpus(input_path, output_path):
    """Clean Russian text in parallel corpus by removing openers and headers"""
    df = pd.read_csv(input_path, sep="\t", encoding="utf-8")

    if "Russian" not in df.columns:
        raise ValueError("Missing 'Russian' column in TSV")

    # Apply opener removal
    df["Russian"] = df["Russian"].astype(str).apply(clean_russian_line)

    # Restore capitalization after opener removal
    df["Russian"] = df["Russian"].apply(restore_initial_cap)

    df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    print(f"Cleaned file saved to: {output_path}")


def restore_initial_cap(text):
    """Restore initial capitalization of Russian text"""
    if not text:
        return text
    return text[0].upper() + text[1:]


if __name__ == "__main__":
    input_file = "data/cleaned_parallel_corpus_initial.tsv"
    output_file = "data/cleaned_parallel_corpus.tsv"
    clean_parallel_corpus(input_file, output_file)

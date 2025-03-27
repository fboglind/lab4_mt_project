"""split_abstracts2.py"""
"""split_abstracts.py - Split parallel abstracts into sentence pairs"""

import re
import argparse
import pandas as pd

# Argument parser
parser = argparse.ArgumentParser(
    description="Split parallel abstracts into sentence pairs"
)
parser.add_argument(
    "--input", required=True, help="Path to the parallel corpus (TSV format)"
)
parser.add_argument(
    "--output", required=True, help="Path to save the sentence-aligned corpus"
)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.input, sep="\t", encoding="utf-8")

# Ensure required columns exist
if "Russian" not in df.columns or "English" not in df.columns:
    raise ValueError(
        "Error: 'Russian' and 'English' columns not found in the input file."
    )

def is_fragment(text):
    """Check if the text is a fragment (short sentence or abbreviation)"""
    text = text.strip()
    # Skip very short texts (less than 3 words or 15 characters)
    if len(text.split()) < 3 or len(text) < 15:
        return True
    # Skip texts that look like abbreviations or references
    if re.search(r"\b(рис\.|табл\.|им\.|[А-Яа-я]\.\s*[А-Яа-я]\.|№\s*\d+|\[?\d+\]?)", text):
        return True
    return False

def split_sentences(text):
    """Improved sentence splitting function for medical abstracts"""
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Common abbreviations in medical texts (Russian and English)
    abbreviations = [
        # Russian
        "г", "гг", "долл", "млн", "млрд", "рис", "табл", "им", "т", 
        "и др", "т.д", "т.п", "см", "о.е", "н.э", "т.е", "ВОЗ", "АРТ",
        "ВИЧ", "СОVID-19", "SARS-CoV-2", "МКБ-10", "F32", "IL-1", "IL-6",
        "IL-8", "IL-10", "TNF-α",
        # English
        "e.g", "i.e", "vs", "fig", "no", "vol", "pp", "al", "et al", 
        "HIV", "ART", "COVID-19", "WHO", "ICD-10"
    ]
    
    # Protect abbreviations
    for abbr in abbreviations:
        text = re.sub(rf"(\s){re.escape(abbr)}\.", rf"\1{abbr}<DOT>", text, flags=re.IGNORECASE)
    
    # Protect decimal numbers and percentages
    text = re.sub(r"(\d+)\.(\d+)", r"\1<DECIMAL>\2", text)
    text = re.sub(r"(\d+)%", r"\1<PERCENT>", text)
    
    # Protect email addresses and URLs
    text = re.sub(r"(\S+@\S+\.\S+)", r"\1<EMAIL>", text)
    text = re.sub(r"(https?://\S+)", r"\1<URL>", text)
    
    # Split sentences - more sophisticated pattern
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-ZА-Я][a-zа-я]\.)(?<=\.|\?|\!|\…)\s+(?=[A-ZА-Я"«])', text)
    
    # Restore protected elements
    for i in range(len(sentences)):
        sentences[i] = sentences[i].replace("<DOT>", ".")
        sentences[i] = re.sub(r"(\d+)<DECIMAL>(\d+)", r"\1.\2", sentences[i])
        sentences[i] = re.sub(r"(\d+)<PERCENT>", r"\1%", sentences[i])
        sentences[i] = sentences[i].replace("<EMAIL>", "")
        sentences[i] = sentences[i].replace("<URL>", "")
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Handle cases where splitting might have gone wrong
    if len(sentences) > 1:
        # Check if the last "sentence" is actually a continuation
        last_sentence = sentences[-1]
        if len(last_sentence.split()) < 5 and not re.search(r"[.!?]$", last_sentence):
            sentences[-2] = sentences[-2] + " " + sentences[-1]
            sentences = sentences[:-1]
    
    return sentences

# Process each abstract into sentences
sentence_pairs = []
for idx, row in df.iterrows():
    ru_text = str(row["Russian"]).strip()
    en_text = str(row["English"]).strip()
    
    if not ru_text or not en_text:
        continue
        
    ru_sentences = split_sentences(ru_text)
    en_sentences = split_sentences(en_text)
    
    print(f"[DEBUG] Abstract {idx}: {len(ru_sentences)} Russian / {len(en_sentences)} English sentences")
    
    # More sophisticated alignment that can handle minor length mismatches
    min_len = min(len(ru_sentences), len(en_sentences))
    max_len = max(len(ru_sentences), len(en_sentences))
    
    # If lengths differ significantly, we might want to handle this differently
    if max_len - min_len > 2:
        print(f"[WARNING] Abstract {idx} has significant sentence count mismatch: {len(ru_sentences)} vs {len(en_sentences)}")
    
    for i in range(min_len):
        if not is_fragment(ru_sentences[i]) and not is_fragment(en_sentences[i]):
            sentence_pairs.append((idx, ru_sentences[i], en_sentences[i]))

# Create new DataFrame with index
sentence_df = pd.DataFrame(sentence_pairs, columns=["AbstractID", "Russian", "English"])

# Save the new sentence-aligned dataset
sentence_df.to_csv(args.output, sep="\t", index=False, encoding='utf-8')
print(f"Sentence-aligned corpus saved to {args.output}")
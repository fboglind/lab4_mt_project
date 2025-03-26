import pandas as pd
import re
from collections import Counter

# Define Russian openers and their regex patterns
openers = {
    #"^Цель": r"^Цель",
    "^Цель исследования": r"^Цель\s+исследования",
    "^Цель работы": r"^Цель\s+работы",
    "^Резюме": r"^Резюме",
    "^Резюме Цель": r"^Резюме\s+Цель",
    "^Резюме Цель исследования": r"^Резюме\s+Цель\s+исследования",
    "^Введение": r"^Введение",
    "^Аннотация": r"^Аннотация"
}

def get_english_openers(file_path, num_words=3):
    df = pd.read_csv(file_path, sep="\t", encoding="utf-8").dropna()
    results = {}

    for label, pattern in openers.items():
        english_openers = []
        match_count = 0
        for _, row in df.iterrows():
            russian = row["Russian"]
            english = row["English"]
            if re.match(pattern, russian, flags=re.IGNORECASE):
                match_count += 1
                words = english.strip().split()
                if words:
                    opener = " ".join(words[:num_words])
                    english_openers.append(opener.lower())
        results[label] = {
            "count": match_count,
            "counter": Counter(english_openers)
        }

    return results

if __name__ == "__main__":
    corpus_path = "data/cleaned_parallel_corpus_initial.tsv"
    opener_stats = get_english_openers(corpus_path, num_words=3)

    for russian_opener, data in opener_stats.items():
        print(f"\n🔹 Russian Opener: {russian_opener}  (Count: {data['count']})")
        for phrase, count in data["counter"].most_common(10):
            print(f"   {phrase:<40} {count}")

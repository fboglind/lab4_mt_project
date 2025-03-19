"""
fix_repetitions.py - Utility to detect and fix repetition issues in translations
"""
import argparse
import logging
import os
import re
from collections import Counter

def setup_logger():
    """Set up logger with appropriate formatting"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def read_lines(file_path):
    """Read lines from a file and return as a list of strings"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def detect_repetitions(text, threshold=3):
    """
    Detect repetitive patterns in text
    
    Args:
        text: Input text
        threshold: Minimum number of repetitions to consider
        
    Returns:
        patterns: Dictionary of patterns and their counts
    """
    words = text.split()
    
    # Look for repeating n-grams (1 to 3 words)
    patterns = {}
    
    for n in range(1, 4):
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        
        for ngram, count in counts.items():
            if count >= threshold:
                pattern = ' '.join(ngram)
                patterns[pattern] = count
    
    return patterns

def fix_repetition(text):
    """
    Fix repetitive patterns in text
    
    Args:
        text: Input text
        
    Returns:
        fixed_text: Text with repetitions fixed
    """
    # Fix simple repeating patterns
    words = text.split()
    
    if len(words) <= 3:
        return text  # Too short to fix
    
    # Find the longest non-repeating prefix
    fixed_words = []
    i = 0
    
    while i < len(words):
        current_word = words[i]
        fixed_words.append(current_word)
        
        # Check for repetition
        repeat_detected = False
        
        # Look for repeating sequences of different lengths
        for length in range(1, min(5, len(fixed_words) + 1)):
            pattern = fixed_words[-length:]
            
            # Check if this pattern repeats next
            if i + length < len(words) and words[i+1:i+length+1] == pattern:
                repeat_detected = True
                i += length  # Skip the repeating pattern
                break
        
        if not repeat_detected:
            i += 1
    
    # Filter out sequences of "the" and "of" and other common patterns
    result = ' '.join(fixed_words)
    
    # Replace repetitive "the of" patterns
    result = re.sub(r'(\bthe\s+of\b\s*){3,}', 'the of ', result)
    
    # Replace repetitive "of the" patterns
    result = re.sub(r'(\bof\s+the\b\s*){3,}', 'of the ', result)
    
    # Replace repetitive sequences of "and"
    result = re.sub(r'(\band\b\s*){3,}', 'and ', result)
    
    # Replace repetitive sequences of "the"
    result = re.sub(r'(\bthe\b\s*){3,}', 'the ', result)
    
    # Replace repetitive sequences of "of"
    result = re.sub(r'(\bof\b\s*){3,}', 'of ', result)
    
    # Replace sequences of comma-separated repeating words
    result = re.sub(r'(,\s*and\b\s*){3,}', ', and ', result)
    
    return result

def process_file(input_file, output_file, logger):
    """
    Process translations file to fix repetitions
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        logger: Logger instance
        
    Returns:
        stats: Dictionary with statistics
    """
    logger.info(f"Reading translations from {input_file}")
    translations = read_lines(input_file)
    
    fixed_translations = []
    stats = {"total": len(translations), "fixed": 0, "repetition_ratio": []}
    
    for i, text in enumerate(translations):
        # Count words in original text
        original_words = text.split()
        original_word_count = len(original_words)
        
        # Get unique words to calculate repetition ratio
        unique_words = len(set(original_words))
        if original_word_count > 0:
            repetition_ratio = unique_words / original_word_count
            stats["repetition_ratio"].append(repetition_ratio)
        else:
            repetition_ratio = 1.0
            stats["repetition_ratio"].append(repetition_ratio)
        
        # Detect if there's a serious repetition issue
        has_repetition = repetition_ratio < 0.3 and original_word_count > 10
        
        # Fix repetitions
        if has_repetition:
            fixed_text = fix_repetition(text)
            fixed_translations.append(fixed_text)
            stats["fixed"] += 1
            
            if (i + 1) % 100 == 0 or has_repetition:
                logger.info(f"Fixed repetition in translation {i+1}: {text[:50]}... -> {fixed_text[:50]}...")
        else:
            fixed_translations.append(text)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed translation {i+1} (no repetition issues)")
    
    # Save fixed translations
    logger.info(f"Writing fixed translations to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for translation in fixed_translations:
            f.write(f"{translation}\n")
    
    # Calculate average repetition ratio
    avg_repetition_ratio = sum(stats["repetition_ratio"]) / len(stats["repetition_ratio"])
    stats["avg_repetition_ratio"] = avg_repetition_ratio
    
    logger.info(f"Fixed {stats['fixed']} out of {stats['total']} translations")
    logger.info(f"Average repetition ratio: {avg_repetition_ratio:.4f}")
    
    return stats

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fix repetition issues in translations")
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to input translations file"
    )
    parser.add_argument(
        "--output", 
        required=True,
        help="Path to output fixed translations file"
    )
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Process file
    stats = process_file(args.input, args.output, logger)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()

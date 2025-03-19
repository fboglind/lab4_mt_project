"""
advanced_evaluation.py - Comprehensive evaluation script for machine translation
"""
import argparse
import logging
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sacrebleu import corpus_bleu, corpus_chrf
from sacrebleu.metrics import BLEU, CHRF
from collections import defaultdict

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
        return [line.strip() for line in f if line.strip()]

def calculate_sentence_bleu(hyp, ref, bleu_scorer):
    """Calculate BLEU score for a single sentence"""
    return bleu_scorer.sentence_score(hyp, [ref]).score

def calculate_sentence_chrf(hyp, ref, chrf_scorer):
    """Calculate chrF score for a single sentence"""
    return chrf_scorer.sentence_score(hyp, [ref]).score

def analyze_length_impact(hypotheses, references, logger):
    """Analyze how sentence length affects translation quality"""
    logger.info("Analyzing impact of sentence length on translation quality...")
    
    # Initialize scorers
    bleu_scorer = BLEU()
    chrf_scorer = CHRF()
    
    # Group sentences by length
    length_bins = defaultdict(list)
    length_bleu = defaultdict(list)
    length_chrf = defaultdict(list)
    
    # Calculate scores for each sentence
    for hyp, ref in zip(hypotheses, references[0]):
        # Calculate reference length (in words)
        ref_length = len(ref.split())
        
        # Calculate scores
        bleu = calculate_sentence_bleu(hyp, ref, bleu_scorer)
        chrf = calculate_sentence_chrf(hyp, ref, chrf_scorer)
        
        # Group by length in bins of 5 words
        length_bin = (ref_length // 5) * 5
        length_bins[length_bin].append(ref_length)
        length_bleu[length_bin].append(bleu)
        length_chrf[length_bin].append(chrf)
    
    # Calculate average scores for each length bin
    bins = sorted(length_bins.keys())
    avg_bleu = [np.mean(length_bleu[b]) for b in bins]
    avg_chrf = [np.mean(length_chrf[b]) for b in bins]
    counts = [len(length_bins[b]) for b in bins]
    
    return bins, avg_bleu, avg_chrf, counts

def create_visualizations(bins, avg_bleu, avg_chrf, counts, output_dir):
    """Create visualizations of evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot BLEU vs sentence length
    plt.figure(figsize=(10, 6))
    plt.bar(bins, avg_bleu, width=4)
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score vs Sentence Length')
    plt.savefig(os.path.join(output_dir, 'bleu_vs_length.png'), dpi=300, bbox_inches='tight')
    
    # Plot chrF vs sentence length
    plt.figure(figsize=(10, 6))
    plt.bar(bins, avg_chrf, width=4)
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('chrF Score')
    plt.title('chrF Score vs Sentence Length')
    plt.savefig(os.path.join(output_dir, 'chrf_vs_length.png'), dpi=300, bbox_inches='tight')
    
    # Plot sentence count distribution
    plt.figure(figsize=(10, 6))
    plt.bar(bins, counts, width=4)
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Number of Sentences')
    plt.title('Distribution of Sentence Lengths')
    plt.savefig(os.path.join(output_dir, 'length_distribution.png'), dpi=300, bbox_inches='tight')

def evaluate_translations(hyp_file, ref_files, output_dir, logger):
    """
    Evaluate translations using BLEU and chrF
    
    Args:
        hyp_file: Path to hypothesis translations
        ref_files: List of paths to reference translations
        output_dir: Directory to save results
        logger: Logger instance
        
    Returns:
        evaluation_results: Dictionary with evaluation results
    """
    logger.info(f"Reading hypothesis translations from {hyp_file}")
    hypotheses = read_lines(hyp_file)
    
    # Read all reference files
    references = []
    for ref_file in ref_files:
        logger.info(f"Reading reference translations from {ref_file}")
        references.append(read_lines(ref_file))
    
    # Ensure all files have the same number of sentences
    min_len = min(len(hypotheses), *[len(refs) for refs in references])
    if len(hypotheses) != min_len:
        logger.warning(f"Truncating hypotheses from {len(hypotheses)} to {min_len} sentences")
        hypotheses = hypotheses[:min_len]
    
    for i, refs in enumerate(references):
        if len(refs) != min_len:
            logger.warning(f"Truncating reference {i+1} from {len(refs)} to {min_len} sentences")
            references[i] = refs[:min_len]
    
    # Calculate BLEU score
    logger.info("Calculating BLEU score...")
    bleu_score = corpus_bleu(hypotheses, references)
    
    # Calculate chrF score
    logger.info("Calculating chrF score...")
    chrf_score = corpus_chrf(hypotheses, references)
    
    # Analyze impact of sentence length
    bins, avg_bleu, avg_chrf, counts = analyze_length_impact(hypotheses, references, logger)
    
    # Create visualizations
    create_visualizations(bins, avg_bleu, avg_chrf, counts, output_dir)
    
    # Compile results
    evaluation_results = {
        'bleu': {
            'score': bleu_score.score,
            'details': str(bleu_score)
        },
        'chrf': {
            'score': chrf_score.score,
            'details': str(chrf_score)
        },
        'length_analysis': {
            'bins': bins,
            'avg_bleu': avg_bleu,
            'avg_chrf': avg_chrf,
            'counts': counts
        }
    }
    
    return evaluation_results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Advanced machine translation evaluation")
    parser.add_argument(
        "--hyp", 
        required=True, 
        help="Path to hypothesis translations"
    )
    parser.add_argument(
        "--ref", 
        required=True, 
        nargs='+',
        help="Path(s) to reference translation(s)"
    )
    parser.add_argument(
        "--output-dir", 
        default="evaluation_results",
        help="Directory to save evaluation results and visualizations"
    )
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger()
    
    # Check if files exist
    for file_path in [args.hyp] + args.ref:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate translations
    results = evaluate_translations(args.hyp, args.ref, args.output_dir, logger)
    
    # Print results
    logger.info(f"BLEU score: {results['bleu']['score']:.2f}")
    logger.info(f"chrF score: {results['chrf']['score']:.4f}")
    
    # Save detailed results to file
    with open(os.path.join(args.output_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write("# Machine Translation Evaluation Results\n\n")
        f.write(f"Hypothesis file: {args.hyp}\n")
        f.write(f"Reference file(s): {', '.join(args.ref)}\n\n")
        f.write("## BLEU Score\n\n")
        f.write(f"BLEU = {results['bleu']['score']:.2f}\n")
        f.write(f"{results['bleu']['details']}\n\n")
        f.write("## chrF Score\n\n")
        f.write(f"chrF = {results['chrf']['score']:.4f}\n")
        f.write(f"{results['chrf']['details']}\n\n")
    
    # Save results as JSON
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

"""evaluate_translations.py - Script for evaluating machine translation quality using sacreBLEU"""

import argparse
import logging
import os
from sacrebleu import corpus_bleu, corpus_chrf


def setup_logger():
    """Set up logger with appropriate formatting"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def read_lines(file_path):
    """Read lines from a file and return as a list of strings"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def evaluate_translations(hyp_file, ref_file, logger):
    """
    Evaluate translations using BLEU and chrF

    Args:
        hyp_file: Path to hypothesis translations (system output)
        ref_file: Path to reference translations (ground truth)
        logger: Logger instance

    Returns:
        bleu_score: BLEU score object
        chrf_score: chrF score object
    """
    logger.info(f"Reading hypothesis translations from {hyp_file}")
    hypotheses = read_lines(hyp_file)

    logger.info(f"Reading reference translations from {ref_file}")
    references = [read_lines(ref_file)]  # sacreBLEU expects a list of reference lists

    # Ensure we have the same number of hypotheses and references
    if len(hypotheses) != len(references[0]):
        logger.warning(
            f"Number of hypotheses ({len(hypotheses)}) doesn't match "
            f"number of references ({len(references[0])})"
        )
        # Truncate to the shorter length
        min_len = min(len(hypotheses), len(references[0]))
        hypotheses = hypotheses[:min_len]
        references[0] = references[0][:min_len]
        logger.warning(f"Truncated to {min_len} sentences for evaluation")

    # Calculate BLEU score
    logger.info("Calculating BLEU score...")
    bleu_score = corpus_bleu(hypotheses, references)

    # Calculate chrF score
    logger.info("Calculating chrF score...")
    chrf_score = corpus_chrf(hypotheses, references)

    return bleu_score, chrf_score


def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate machine translation using sacreBLEU"
    )
    parser.add_argument(
        "--hyp", required=True, help="Path to hypothesis translations (system output)"
    )
    parser.add_argument(
        "--ref", required=True, help="Path to reference translations (ground truth)"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.txt",
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    # Set up logger
    logger = setup_logger()

    # Check if files exist
    for file_path in [args.hyp, args.ref]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

    # Evaluate translations
    bleu_score, chrf_score = evaluate_translations(args.hyp, args.ref, logger)

    # Print results
    logger.info(f"BLEU score: {bleu_score.score:.2f}")
    logger.info(f"chrF score: {chrf_score.score:.4f}")

    # Save detailed results to file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# Machine Translation Evaluation Results\n\n")
        f.write(f"Hypothesis file: {args.hyp}\n")
        f.write(f"Reference file: {args.ref}\n\n")
        f.write("## BLEU Score\n\n")
        f.write(f"BLEU = {bleu_score.score:.2f}\n")
        f.write(f"{bleu_score}\n\n")
        f.write("## chrF Score\n\n")
        f.write(f"chrF = {chrf_score.score:.4f}\n")
        f.write(f"{chrf_score}\n\n")

    logger.info(f"Detailed evaluation results saved to {args.output}")


if __name__ == "__main__":
    main()

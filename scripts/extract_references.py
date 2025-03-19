"""
extract_references.py - Utility script to extract reference translations from parallel corpus
"""
import argparse
import pandas as pd
import logging
import os

def setup_logger():
    """Set up logger with appropriate formatting"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def extract_references(parallel_file, test_ids_file, output_file, src_col, tgt_col, logger):
    """
    Extract reference translations for test sentences
    
    Args:
        parallel_file: Path to the parallel corpus TSV file
        test_ids_file: Path to the file containing test sentence IDs or test sentences
        output_file: Path to save reference translations
        src_col: Column name for source language
        tgt_col: Column name for target language
        logger: Logger instance
        
    Returns:
        success: Boolean indicating success
    """
    try:
        logger.info(f"Loading parallel corpus from {parallel_file}")
        df = pd.read_csv(parallel_file, sep='\t', encoding='utf-8')
        
        if src_col not in df.columns or tgt_col not in df.columns:
            logger.error(f"Columns not found in parallel corpus: {src_col}, {tgt_col}")
            return False
        
        logger.info(f"Reading test sentences from {test_ids_file}")
        with open(test_ids_file, 'r', encoding='utf-8') as f:
            test_sentences = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Found {len(test_sentences)} test sentences")
        
        # Find matching source sentences in the parallel corpus
        references = []
        found_count = 0
        
        for test_sent in test_sentences:
            # Try to find an exact match
            matches = df[df[src_col] == test_sent]
            
            if not matches.empty:
                # Take the first match
                references.append(matches.iloc[0][tgt_col])
                found_count += 1
            else:
                # If no exact match, add an empty placeholder
                references.append("")
                logger.warning(f"No match found for test sentence: {test_sent[:50]}...")
        
        logger.info(f"Found references for {found_count} out of {len(test_sentences)} test sentences")
        
        # Save references to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for ref in references:
                f.write(f"{ref}\n")
        
        logger.info(f"Reference translations saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting references: {str(e)}")
        return False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract reference translations from parallel corpus")
    parser.add_argument(
        "--parallel", 
        required=True, 
        help="Path to the parallel corpus TSV file"
    )
    parser.add_argument(
        "--test", 
        required=True,
        help="Path to the file containing test sentences"
    )
    parser.add_argument(
        "--output", 
        required=True,
        help="Path to save reference translations"
    )
    parser.add_argument(
        "--src-col", 
        default="Russian",
        help="Column name for source language in parallel corpus"
    )
    parser.add_argument(
        "--tgt-col", 
        default="English",
        help="Column name for target language in parallel corpus"
    )
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger()
    
    # Check if files exist
    if not os.path.exists(args.parallel):
        logger.error(f"Parallel corpus file not found: {args.parallel}")
        return
    
    if not os.path.exists(args.test):
        logger.error(f"Test sentences file not found: {args.test}")
        return
    
    # Extract references
    success = extract_references(
        args.parallel, args.test, args.output, 
        args.src_col, args.tgt_col, logger
    )
    
    if success:
        logger.info("Reference extraction completed successfully")
    else:
        logger.error("Reference extraction failed")

if __name__ == "__main__":
    main()

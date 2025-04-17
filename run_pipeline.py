import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/raw/EARC Dataset/Reference',  # For training data
        'data/raw/EARC Dataset/Papers',     # For papers to evaluate
        'data/processed',
        'models',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_reference_files():
    """Check if reference PDF files exist for training."""
    ref_dir = 'data/raw/EARC Dataset/Reference'
    
    # Check for Publishable and Non-Publishable directories
    publishable_dir = Path(ref_dir) / 'Publishable'
    non_publishable_dir = Path(ref_dir) / 'Non-Publishable'
    
    if not (publishable_dir.exists() and non_publishable_dir.exists()):
        logger.error(f"Reference directories not found in {ref_dir}")
        logger.error("Please ensure both 'Publishable' and 'Non-Publishable' directories exist")
        return False
    
    # Count PDF files in both directories
    publishable_files = list(publishable_dir.glob('*.pdf'))
    non_publishable_files = list(non_publishable_dir.glob('*.pdf'))
    
    if not (publishable_files and non_publishable_files):
        logger.error("No PDF files found in reference directories")
        logger.error("Please add reference PDF files before training")
        return False
    
    logger.info(f"Found {len(publishable_files)} publishable and {len(non_publishable_files)} non-publishable reference papers")
    return True

def check_papers_to_evaluate():
    """Check if PDF files exist in the Papers directory for evaluation."""
    papers_dir = 'data/raw/EARC Dataset/Papers'
    pdf_files = list(Path(papers_dir).glob('*.pdf'))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {papers_dir}")
        logger.error("Please add papers to evaluate before running the pipeline")
        return False
    
    logger.info(f"Found {len(pdf_files)} papers to evaluate")
    return True

def main():
    # Step 1: Setup directories
    logger.info("Setting up directories...")
    setup_directories()
    
    # Step 2: Check for reference files (training data)
    logger.info("\nChecking reference files for training...")
    if not check_reference_files():
        return
    
    # Step 3: Check for papers to evaluate
    logger.info("\nChecking papers to evaluate...")
    if not check_papers_to_evaluate():
        return
    
    # Step 4: Process reference papers and train the model
    logger.info("\nProcessing reference papers and training model...")
    try:
        from src.preprocess import process_earc_dataset
        from src.model_train import train_and_evaluate
        
        # Process reference papers
        logger.info("Processing reference papers...")
        df = process_earc_dataset(base_path="data/raw/EARC Dataset/Reference")
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        df.to_pickle('data/processed/processed_papers.pkl')
        logger.info("Reference papers processed and saved")
        
        # Train model
        logger.info("Training model...")
        results = train_and_evaluate('data/processed/processed_papers.pkl')
        logger.info("Model training completed successfully")
        logger.info(f"Training accuracy: {results['metrics']['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Error during reference processing and training: {str(e)}")
        return
    
    # Step 5: Generate results for papers to evaluate
    logger.info("\nGenerating results for papers to evaluate...")
    try:
        from src.generate_results import generate_results
        results_df = generate_results(data_dir="data/raw/EARC Dataset/Papers")
        logger.info("Results generation completed successfully")
    except Exception as e:
        logger.error(f"Error during results generation: {str(e)}")
        return
    
    logger.info("\nPipeline completed successfully!")
    logger.info("Check results/results.csv for the final output")

if __name__ == "__main__":
    main() 
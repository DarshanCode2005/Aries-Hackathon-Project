import os
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup():
    """Remove all generated files and directories to start fresh."""
    
    # Files and directories to remove
    to_remove = [
        'data/processed/processed_papers.pkl',
        'data/processed/results.csv',
        'results/results.csv',
        'models/classifier.joblib',
        'models/label_encoder.joblib',
        'models/training_results.json'
    ]
    
    # Remove individual files
    for file_path in to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed: {file_path}")
            except Exception as e:
                logger.error(f"Error removing {file_path}: {str(e)}")
    
    # Clean empty directories
    directories = ['data/processed', 'models', 'results']
    for directory in directories:
        if os.path.exists(directory) and not os.listdir(directory):
            try:
                os.rmdir(directory)
                logger.info(f"Removed empty directory: {directory}")
            except Exception as e:
                logger.error(f"Error removing directory {directory}: {str(e)}")
    
    logger.info("Cleanup completed!")

if __name__ == "__main__":
    cleanup() 
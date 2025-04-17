import pandas as pd
from qa_system import RAGQASystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_qa_index(data_path: str = "data/processed/processed_papers.pkl") -> None:
    """
    Build the FAISS index for the Q&A system from processed papers.
    
    Args:
        data_path (str): Path to the processed papers DataFrame
    """
    try:
        # Load processed papers
        logger.info(f"Loading processed papers from {data_path}")
        df = pd.read_pickle(data_path)
        
        # Initialize QA system
        qa_system = RAGQASystem()
        
        # Build index from paper texts
        logger.info("Building FAISS index...")
        qa_system.build_index(df['Cleaned_Text'].tolist())
        
        logger.info("Index built successfully!")
        
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        raise

if __name__ == "__main__":
    build_qa_index() 
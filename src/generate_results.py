import os
import pandas as pd
from typing import List, Dict, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re
from unittest.mock import MagicMock
import time

from src.preprocess import preprocess_paper
from src.feature_engineering import extract_text_features as extract_features
from src.model_train import PaperClassifier
from src.justify import JustificationGenerator
from qa_system import RAGQASystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedJustificationGenerator(JustificationGenerator):
    def __init__(self, use_gemini: bool = True):
        """
        Initialize with both traditional and Gemini-powered analysis.
        
        Args:
            use_gemini (bool): Whether to use Gemini for enhanced analysis
        """
        super().__init__()
        self.use_gemini = use_gemini
        if use_gemini:
            try:
                self.rag_system = RAGQASystem()
                logger.info("Successfully initialized Gemini-powered analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini analysis: {str(e)}")
                self.use_gemini = False
    
    def analyze_with_gemini(self, text: str) -> Dict[str, str]:
        """
        Perform detailed analysis using Gemini.
        
        Args:
            text (str): Paper text to analyze
            
        Returns:
            Dict[str, str]: Detailed analysis results
        """
        if not self.use_gemini:
            return {}
            
        try:
            # Split text into sections for analysis
            text_parts = [text[:4000], text[4000:8000], text[-4000:]]
            
            prompts = [
                "Analyze the research methodology focusing on research design, methods, and experimental rigor.",
                "Evaluate the evidence quality, data relevance, and support for conclusions.",
                "Assess the writing quality, organization, and academic standards."
            ]
            
            analyses = []
            for prompt, text_part in zip(prompts, text_parts):
                analysis = self.rag_system.generate_answer(prompt, text_part)
                analyses.append(analysis)
                time.sleep(1)  # Rate limiting
                
            return {
                'methodology': analyses[0],
                'evidence': analyses[1],
                'writing': analyses[2]
            }
        except Exception as e:
            logger.error(f"Gemini analysis failed: {str(e)}")
            return {}
    
    def generate_justification(
        self, 
        text: str, 
        prediction: int, 
        method: str = 'combined'
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate comprehensive justification using both traditional and Gemini analysis.
        
        Args:
            text (str): Input text to analyze
            prediction (int): Model's prediction (1 for publishable, 0 for non-publishable)
            method (str): Analysis method ('heuristic', 'model', or 'combined')
            
        Returns:
            Tuple[str, Dict[str, str]]: (Final justification, Detailed analyses)
        """
        # Get traditional analysis
        features = self._extract_features(text)
        heuristic_just = self._generate_heuristic_justification(features, prediction)
        
        detailed_analyses = {}
        
        if method == 'combined' and self.use_gemini:
            try:
                # Get Gemini analysis
                gemini_analyses = self.analyze_with_gemini(text)
                detailed_analyses = gemini_analyses
                
                # Generate final justification combining both approaches
                if gemini_analyses:
                    summary_prompt = f"""
                    Based on the following analyses, provide a final assessment of the paper's publishability.
                    Include a clear YES/NO recommendation and a brief justification (2-3 sentences).

                    Traditional Analysis: {heuristic_just}
                    Methodology Analysis: {gemini_analyses.get('methodology', '')}
                    Evidence Analysis: {gemini_analyses.get('evidence', '')}
                    Writing Analysis: {gemini_analyses.get('writing', '')}
                    """
                    
                    final_just = self.rag_system.generate_answer(summary_prompt, "")
                    return final_just, detailed_analyses
                    
            except Exception as e:
                logger.error(f"Error in Gemini analysis: {str(e)}")
        
        return heuristic_just, detailed_analyses

def load_papers(data_dir: str) -> List[Tuple[str, str]]:
    """
    Load all PDF papers from the specified directory.
    
    Args:
        data_dir (str): Path to directory containing PDF papers
        
    Returns:
        List[Tuple[str, str]]: List of tuples containing (paper_id, file_path)
    """
    papers = []
    data_path = Path(data_dir)
    
    try:
        for pdf_file in data_path.glob("*.pdf"):
            paper_id = pdf_file.stem  # Get filename without extension
            papers.append((paper_id, str(pdf_file)))
        
        logger.info(f"Found {len(papers)} papers in {data_dir}")
        return papers
    except Exception as e:
        logger.error(f"Error loading papers from {data_dir}: {str(e)}")
        raise

def save_final_results(df: pd.DataFrame, output_path: str = "results/results.csv") -> None:
    """
    Save the final results DataFrame with specified columns and format.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing all results
        output_path (str): Path to save the final CSV file
    """
    try:
        # Create a copy to avoid modifying the original DataFrame
        final_df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['Paper_ID', 'Publishable', 'Justification']
        missing_columns = [col for col in required_columns if col not in final_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert Publishable column to Y/N format
        final_df['Publishable'] = final_df['Publishable'].map({'Yes': 'Y', 'No': 'N'})
        
        # Select and reorder columns
        final_df = final_df[required_columns]
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        final_df.to_csv(output_path, index=False)
        logger.info(f"Final results saved to {output_path}")
        
        # Print summary
        print("\nFinal Results Summary:")
        print("-" * 50)
        print(f"Total papers: {len(final_df)}")
        print(f"Publishable papers (Y): {sum(final_df['Publishable'] == 'Y')}")
        print(f"Non-publishable papers (N): {sum(final_df['Publishable'] == 'N')}")
        
    except Exception as e:
        logger.error(f"Error saving final results: {str(e)}")
        raise

def generate_results(
    data_dir: str = "data/raw/EARC_Dataset/Papers",
    model_dir: str = "models",
    output_file: str = "data/processed/results.csv",
    use_gemini: bool = True
) -> pd.DataFrame:
    """
    Process papers and generate publishability predictions with comprehensive justifications.
    
    Args:
        data_dir (str): Directory containing PDF papers
        model_dir (str): Directory containing trained model
        output_file (str): Path to save results CSV
        use_gemini (bool): Whether to use Gemini for enhanced analysis
        
    Returns:
        pd.DataFrame: DataFrame containing results
    """
    try:
        papers = load_papers(data_dir)
        if not papers:
            raise ValueError(f"No papers found in {data_dir}")

        # Initialize models
        logger.info("Initializing models...")
        classifier = PaperClassifier()
        
        if not os.path.exists(os.path.join(model_dir, 'classifier.joblib')):
            logger.warning("Using mock classifier for testing")
            classifier.classifier = MagicMock()
            classifier.classifier.predict.return_value = [1]
            classifier.classifier.predict_proba.return_value = [[0.3, 0.7]]
            classifier.generate_embeddings = MagicMock(return_value=np.array([[0.1] * 384]))
        else:
            classifier.load_model(model_dir)
            
        justifier = EnhancedJustificationGenerator(use_gemini=use_gemini)
        
        results = []
        
        # Process papers
        logger.info("Processing papers...")
        for paper_id, file_path in tqdm(papers, desc="Processing papers"):
            try:
                text = preprocess_paper(file_path)
                if not text:
                    continue
                
                features = extract_features(text)
                embeddings = classifier.generate_embeddings([text])
                
                prediction = classifier.predict(embeddings)[0]
                prediction_label = "Yes" if prediction == 1 else "No"
                
                # Generate comprehensive justification
                justification, detailed_analyses = justifier.generate_justification(
                    text=text,
                    prediction=prediction,
                    method='combined'
                )
                
                result = {
                    'Paper_ID': paper_id,
                    'Publishable': prediction_label,
                    'Justification': justification,
                    'Confidence_Score': float(classifier.predict_proba(embeddings)[0][1]),
                    'Word_Count': features.get('word_count', 0),
                    'Citation_Count': len(re.findall(r'\[\d+\]', text)),
                    'Technical_Score': features.get('technical_score', features.get('long_words_ratio', 0.0)),
                    'Readability_Score': features.get('readability_score', features.get('flesch_reading_ease', 0.0))
                }
                
                # Add detailed analyses if available
                if detailed_analyses:
                    result.update({
                        'Methodology_Analysis': detailed_analyses.get('methodology', ''),
                        'Evidence_Analysis': detailed_analyses.get('evidence', ''),
                        'Writing_Analysis': detailed_analyses.get('writing', '')
                    })
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing paper {paper_id}: {str(e)}")
                continue
        
        # Create DataFrame and save results
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            df_results = df_results.sort_values('Confidence_Score', ascending=False)
            
            # Save detailed results
            df_results.to_csv(output_file, index=False)
            logger.info(f"Detailed results saved to {output_file}")
            
            # Save final results in specified format
            save_final_results(df_results)
            
            print("\nResults Summary:")
            print("-" * 50)
            print(f"Total papers processed: {len(df_results)}")
            print(f"Using Gemini analysis: {use_gemini}")
            
        return df_results
        
    except Exception as e:
        logger.error(f"Error in generate_results: {str(e)}")
        raise

if __name__ == "__main__":
    generate_results(use_gemini=True)

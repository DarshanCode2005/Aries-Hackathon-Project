import os
import pandas as pd
from qa_system import RAGQASystem
from preprocess import preprocess_paper
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperJustificationGenerator:
    def __init__(self):
        """Initialize the justification generator with RAG QA system."""
        self.qa_system = RAGQASystem()
        
    def analyze_paper(self, paper_text: str) -> Dict[str, str]:
        """
        Analyze a paper using Gemini to generate detailed justification.
        
        Args:
            paper_text (str): Preprocessed text of the paper
            
        Returns:
            Dict[str, str]: Dictionary containing analysis results
        """
        # Prepare prompts for different aspects of analysis
        methodology_prompt = """
        Analyze the research methodology of this paper and provide a brief assessment (2-3 sentences) focusing on:
        - Research design and approach
        - Data collection and analysis methods
        - Experimental rigor and validity
        
        Paper text:
        {text}
        """.format(text=paper_text[:4000])  # Using first 4000 chars for methodology
        
        evidence_prompt = """
        Evaluate the evidence and results presented in this paper (2-3 sentences):
        - Quality and relevance of data
        - Statistical significance and validity
        - Support for conclusions
        
        Paper text:
        {text}
        """.format(text=paper_text[4000:8000])  # Using next 4000 chars for evidence
        
        writing_prompt = """
        Assess the writing quality and structure of this paper (2-3 sentences):
        - Clarity and organization
        - Academic writing standards
        - Technical communication
        
        Paper text:
        {text}
        """.format(text=paper_text[-4000:])  # Using last 4000 chars for writing
        
        try:
            # Get analysis for each aspect
            methodology_analysis = self.qa_system.generate_answer(methodology_prompt, paper_text[:4000])
            time.sleep(1)  # Rate limiting
            evidence_analysis = self.qa_system.generate_answer(evidence_prompt, paper_text[4000:8000])
            time.sleep(1)  # Rate limiting
            writing_analysis = self.qa_system.generate_answer(writing_prompt, paper_text[-4000:])
            
            return {
                'methodology_analysis': methodology_analysis,
                'evidence_analysis': evidence_analysis,
                'writing_analysis': writing_analysis
            }
        except Exception as e:
            logger.error(f"Error in paper analysis: {str(e)}")
            return {
                'methodology_analysis': f"Error: {str(e)}",
                'evidence_analysis': f"Error: {str(e)}",
                'writing_analysis': f"Error: {str(e)}"
            }
    
    def generate_final_justification(self, analyses: Dict[str, str]) -> Dict[str, any]:
        """
        Generate final justification and publishability assessment based on analyses.
        
        Args:
            analyses (Dict[str, str]): Dictionary containing different analyses
            
        Returns:
            Dict[str, any]: Final assessment including justification and publishability
        """
        summary_prompt = f"""
        Based on the following analyses of a research paper, provide a final assessment of its publishability.
        Include a clear YES/NO recommendation and a brief justification (2-3 sentences).

        Methodology Analysis: {analyses['methodology_analysis']}
        Evidence Analysis: {analyses['evidence_analysis']}
        Writing Analysis: {analyses['writing_analysis']}
        """
        
        try:
            final_assessment = self.qa_system.generate_answer(summary_prompt, "")
            
            # Extract publishability recommendation (YES/NO)
            is_publishable = "YES" in final_assessment.upper()[:50]  # Check first 50 chars
            
            return {
                'is_publishable': is_publishable,
                'justification': final_assessment
            }
        except Exception as e:
            logger.error(f"Error generating final justification: {str(e)}")
            return {
                'is_publishable': None,
                'justification': f"Error: {str(e)}"
            }

def main():
    # Create output directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Read existing results
    try:
        results_df = pd.read_csv('results/results.csv')
    except Exception as e:
        logger.error(f"Error reading results.csv: {str(e)}")
        return
    
    # Initialize generator
    generator = PaperJustificationGenerator()
    
    # New results list
    new_results = []
    
    # Process each paper
    for index, row in results_df.iterrows():
        paper_id = row['Paper_ID']
        logger.info(f"Processing paper {paper_id}")
        
        try:
            # Preprocess paper
            paper_text = preprocess_paper(f"data/papers/{paper_id}.pdf")
            
            # Generate analyses
            analyses = generator.analyze_paper(paper_text)
            
            # Generate final assessment
            final_assessment = generator.generate_final_justification(analyses)
            
            # Store results
            new_results.append({
                'Paper_ID': paper_id,
                'Original_Label': row['Publishable'],
                'Gemini_Label': final_assessment['is_publishable'],
                'Methodology_Analysis': analyses['methodology_analysis'],
                'Evidence_Analysis': analyses['evidence_analysis'],
                'Writing_Analysis': analyses['writing_analysis'],
                'Final_Justification': final_assessment['justification']
            })
            
            logger.info(f"Completed analysis for {paper_id}")
            
        except Exception as e:
            logger.error(f"Error processing {paper_id}: {str(e)}")
            new_results.append({
                'Paper_ID': paper_id,
                'Original_Label': row['Publishable'],
                'Gemini_Label': None,
                'Methodology_Analysis': f"Error: {str(e)}",
                'Evidence_Analysis': f"Error: {str(e)}",
                'Writing_Analysis': f"Error: {str(e)}",
                'Final_Justification': f"Error: {str(e)}"
            })
        
        # Add delay between papers to respect rate limits
        time.sleep(2)
    
    # Create new DataFrame
    new_df = pd.DataFrame(new_results)
    
    # Save results
    output_path = 'results/generated_justifications.csv'
    new_df.to_csv(output_path, index=False)
    logger.info(f"Saved generated justifications to {output_path}")
    
    # Print summary
    total_papers = len(new_results)
    successful = sum(1 for r in new_results if 'Error' not in r['Final_Justification'])
    logger.info(f"Processed {successful}/{total_papers} papers successfully")

if __name__ == "__main__":
    main() 
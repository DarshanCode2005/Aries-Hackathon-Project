import torch
from transformers import AutoTokenizer, pipeline
from typing import Dict, List, Union
import numpy as np
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JustificationGenerator:
    def __init__(self, model_name: str = 'facebook/bart-large-cnn'):
        """
        Initialize the justification generator with a summarization model.
        
        Args:
            model_name (str): Name of the pretrained model to use
        """
        try:
            self.summarizer = pipeline("summarization", model=model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Keywords for heuristic analysis
            self.methodology_keywords = [
                'methodology', 'method', 'approach', 'experiment', 'analysis',
                'procedure', 'design', 'framework', 'technique', 'protocol'
            ]
            self.evidence_keywords = [
                'evidence', 'result', 'finding', 'data', 'observation',
                'outcome', 'measurement', 'statistical', 'significance', 'proof'
            ]
            self.coherence_keywords = [
                'coherent', 'consistent', 'structured', 'organized', 'logical',
                'flow', 'clear', 'cohesive', 'systematic', 'well-presented'
            ]
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract features from the text for heuristic analysis.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, float]: Dictionary of extracted features
        """
        text_lower = text.lower()
        
        # Calculate keyword presence scores with more stringent thresholds
        methodology_score = sum(1 for kw in self.methodology_keywords if kw in text_lower) / len(self.methodology_keywords)
        evidence_score = sum(1 for kw in self.evidence_keywords if kw in text_lower) / len(self.evidence_keywords)
        coherence_score = sum(1 for kw in self.coherence_keywords if kw in text_lower) / len(self.coherence_keywords)
        
        # More detailed analysis
        has_numbers = bool(re.search(r'\d+(?:\.\d+)?%?', text))
        has_citations = bool(re.search(r'\[\d+\]|\(\w+\s*,\s*\d{4}\)', text))
        has_methodology_section = bool(re.search(r'method(ology)?|approach|procedure', text_lower))
        has_results_section = bool(re.search(r'results?|findings?|analysis', text_lower))
        has_discussion = bool(re.search(r'discuss(ion)?|conclusion|implications?', text_lower))
        
        # Check for red flags
        has_plagiarism_markers = bool(re.search(r'copied|plagiari[sz]ed|duplicate', text_lower))
        has_poor_grammar = len(re.findall(r'[.!?]\s+[a-z]', text)) > 5  # Sentences starting with lowercase
        has_formatting_issues = len(re.findall(r'\s{2,}|\t{2,}', text)) > 10  # Excessive spacing
        
        return {
            'methodology_score': methodology_score,
            'evidence_score': evidence_score,
            'coherence_score': coherence_score,
            'has_numbers': has_numbers,
            'has_citations': has_citations,
            'has_methodology_section': has_methodology_section,
            'has_results_section': has_results_section,
            'has_discussion': has_discussion,
            'has_plagiarism_markers': has_plagiarism_markers,
            'has_poor_grammar': has_poor_grammar,
            'has_formatting_issues': has_formatting_issues
        }
    
    def _generate_heuristic_justification(self, features: Dict[str, float], prediction: int) -> str:
        """
        Generate justification based on heuristic analysis with stricter criteria.
        
        Args:
            features (Dict[str, float]): Extracted features
            prediction (int): Model's prediction (1 for publishable, 0 for non-publishable)
            
        Returns:
            str: Generated justification
        """
        # Define stricter thresholds
        METHODOLOGY_THRESHOLD = 0.4
        EVIDENCE_THRESHOLD = 0.4
        COHERENCE_THRESHOLD = 0.4
        
        # Evaluate quality metrics
        methodology_quality = "Strong" if features['methodology_score'] > METHODOLOGY_THRESHOLD else "Weak"
        evidence_quality = "Well-supported" if (features['evidence_score'] > EVIDENCE_THRESHOLD and features['has_numbers']) else "Limited"
        coherence_quality = "Clear and coherent" if features['coherence_score'] > COHERENCE_THRESHOLD else "Needs improvement"
        
        # Check for critical issues
        critical_issues = []
        if features['has_plagiarism_markers']:
            critical_issues.append("potential plagiarism concerns")
        if not features['has_methodology_section']:
            critical_issues.append("missing methodology section")
        if not features['has_results_section']:
            critical_issues.append("missing results section")
        if features['has_poor_grammar']:
            critical_issues.append("significant grammar issues")
        if not features['has_citations']:
            critical_issues.append("lack of citations")
        
        # Generate justification
        justification = []
        
        if prediction == 0 or critical_issues:
            justification.append("Non-publishable due to:")
            if critical_issues:
                justification.append(", ".join(critical_issues))
            justification.append(f"Paper shows {methodology_quality.lower()} methodology with {evidence_quality.lower()} evidence.")
        else:
            justification.append(f"Publishable paper with {methodology_quality.lower()} methodology.")
            justification.append(f"{evidence_quality} by data and analysis.")
            if features['has_citations']:
                justification.append("Appropriate citations present.")
        
        if features['has_formatting_issues']:
            justification.append("Note: Some formatting issues need attention.")
        
        return " ".join(justification).strip()
    
    def _generate_model_summary(self, text: str, max_length: int = 100) -> str:
        """
        Generate a summary using the summarization model.
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum length of the summary
            
        Returns:
            str: Generated summary
        """
        try:
            # The pipeline returns a list of dictionaries
            result = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=30, 
                do_sample=False
            )
            if result and isinstance(result, list) and len(result) > 0:
                return result[0].get('summary_text', 'No summary generated.')
            return 'No summary generated.'
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate summary."
    
    def generate_justification(
        self, 
        text: str, 
        prediction: int, 
        method: str = 'heuristic'
    ) -> str:
        """
        Generate a structured justification for the classification.
        
        Args:
            text (str): Input text to analyze
            prediction (int): Model's prediction (1 for publishable, 0 for non-publishable)
            method (str): Method to use ('heuristic' or 'model')
            
        Returns:
            str: Generated justification
        """
        try:
            # Default to heuristic method if model fails
            features = self._extract_features(text)
            heuristic_justification = self._generate_heuristic_justification(features, prediction)
            
            if method == 'model':
                try:
                    # Prepare a clear prompt for the model
                    prompt = f"""Analyze this research paper and explain why it is {'publishable' if prediction == 1 else 'not publishable'}.
                    Focus on: methodology, evidence quality, and writing clarity.
                    Keep the explanation concise (2-3 sentences).
                    
                    Text excerpt: {text[:1500]}  # Limit text length to avoid token limits
                    """
                    
                    summary = self._generate_model_summary(prompt)
                    if summary and summary != 'No summary generated.' and summary != 'Unable to generate summary.':
                        return summary
                    
                    # Fall back to heuristic if model fails
                    logger.warning("Model-based justification failed, using heuristic method")
                    return heuristic_justification
                    
                except Exception as e:
                    logger.warning(f"Error in model-based justification: {str(e)}, falling back to heuristic")
                    return heuristic_justification
            
            return heuristic_justification
            
        except Exception as e:
            logger.error(f"Error generating justification: {str(e)}")
            return "Unable to generate justification due to an error."

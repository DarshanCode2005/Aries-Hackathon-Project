import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import textstat
import re
from typing import Dict

# Download required NLTK data
nltk.download('punkt')

def extract_text_features(text: str) -> Dict:
    """
    Extract features from cleaned text including word count, sentence metrics,
    readability scores, and presence of key sections.
    
    Args:
        text (str): Cleaned text to analyze
        
    Returns:
        Dict: Dictionary containing extracted features
    """
    # Initialize features dictionary
    features = {}
    
    # Basic text metrics
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    features['word_count'] = len(words)
    features['sentence_count'] = len(sentences)
    
    # Average sentence length
    if len(sentences) > 0:
        features['avg_sentence_length'] = len(words) / len(sentences)
    else:
        features['avg_sentence_length'] = 0
        
    # Readability scores
    features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
    features['gunning_fog_index'] = textstat.gunning_fog(text)
    
    # Section presence detection (case-insensitive)
    section_patterns = {
        'has_methodology': r'\b(methodology|methods|experimental setup)\b',
        'has_results': r'\b(results|findings|observations)\b',
        'has_conclusion': r'\b(conclusion|conclusions|summary|final remarks)\b'
    }
    
    # Check for presence of each section
    for feature_name, pattern in section_patterns.items():
        features[feature_name] = bool(re.search(pattern, text, re.IGNORECASE))
    
    # Additional features
    features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
    
    # Fix: Count only words longer than 6 characters
    long_words = [word for word in words if len(word) > 6]
    features['long_words_ratio'] = len(long_words) / len(words) if words else 0
    
    return features

def get_section_text(text: str, section_name: str) -> str:
    """
    Extract text from a specific section of the paper.
    
    Args:
        text (str): Full paper text
        section_name (str): Name of the section to extract
        
    Returns:
        str: Text from the specified section, or empty string if not found
    """
    # Common section headers (add more variations as needed)
    section_patterns = {
        'abstract': r'abstract',
        'introduction': r'introduction',
        'methodology': r'(methodology|methods|experimental setup)',
        'results': r'(results|findings)',
        'discussion': r'discussion',
        'conclusion': r'(conclusion|conclusions|summary|final remarks)'
    }
    
    if section_name.lower() not in section_patterns:
        raise ValueError(f"Unknown section: {section_name}")
    
    pattern = section_patterns[section_name.lower()]
    
    # Fix: Make the section header pattern more flexible
    section_start = re.compile(f"^\\s*{pattern}\\s*$", re.IGNORECASE | re.MULTILINE)
    match = section_start.search(text)
    
    if not match:
        return ""
    
    start_pos = match.end()
    
    # Find the next section (if any)
    next_section_pattern = '|'.join(section_patterns.values())
    next_section = re.compile(f"^\\s*({next_section_pattern})\\s*$", re.IGNORECASE | re.MULTILINE)
    next_match = next_section.search(text, start_pos)
    
    if next_match:
        end_pos = next_match.start()
        return text[start_pos:end_pos].strip()
    else:
        return text[start_pos:].strip()

def calculate_section_features(text: str) -> Dict:
    """
    Calculate features for each major section of the paper.
    
    Args:
        text (str): Full paper text
        
    Returns:
        Dict: Dictionary containing features for each section
    """
    sections = ['abstract', 'introduction', 'methodology', 'results', 'discussion', 'conclusion']
    features = {}
    
    for section in sections:
        section_text = get_section_text(text, section)
        if section_text:
            section_features = extract_text_features(section_text)
            features[f"{section}_features"] = section_features
    
    return features

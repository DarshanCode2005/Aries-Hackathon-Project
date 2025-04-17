import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import textstat
import re
from typing import Dict
from collections import Counter
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def detect_quality_issues(text: str) -> Dict:
    """
    Detect various quality issues in the text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Dict: Dictionary containing quality metrics
    """
    issues = {}
    
    # 1. Check for repeated phrases (potential copy-paste)
    sentences = sent_tokenize(text.lower())
    trigrams = []
    for sent in sentences:
        words = word_tokenize(sent)
        trigrams.extend(list(ngrams(words, 3)))
    
    trigram_counts = Counter(trigrams)
    repeated_phrases = {' '.join(phrase): count for phrase, count in trigram_counts.items() if count > 2}
    issues['repeated_phrases'] = repeated_phrases
    issues['has_excessive_repetition'] = len(repeated_phrases) > 0
    
    # 2. Check for poor sentence structure
    issues['poor_sentence_count'] = sum(1 for s in sentences if len(s.split()) < 5 or len(s.split()) > 40)
    issues['poor_sentence_ratio'] = issues['poor_sentence_count'] / len(sentences) if sentences else 0
    
    # 3. Check for informal language
    informal_words = {'thing', 'stuff', 'kind of', 'sort of', 'basically', 'pretty much', 'a lot'}
    words = word_tokenize(text.lower())
    informal_count = sum(1 for word in words if word in informal_words)
    issues['informal_language_ratio'] = informal_count / len(words) if words else 0
    
    # 4. Check citation patterns
    citation_pattern = r'\[\d+\]|\(\w+\s*,\s*\d{4}\)'
    citations = re.findall(citation_pattern, text)
    issues['citation_count'] = len(citations)
    issues['citations_per_paragraph'] = len(citations) / (text.count('\n\n') + 1)
    
    # 5. Check for mathematical/technical content
    math_symbols = r'[+\-*/=<>≤≥∈∀∃∑∏∫√π]'
    equations = re.findall(r'\$.*?\$|\\\[.*?\\\]', text)  # LaTeX equations
    issues['has_equations'] = len(equations) > 0
    issues['math_symbol_count'] = len(re.findall(math_symbols, text))
    
    # 6. Check for methodology indicators
    methodology_keywords = {
        'experiment', 'method', 'procedure', 'analysis', 'data', 'sample',
        'measurement', 'algorithm', 'technique', 'methodology'
    }
    methods_count = sum(1 for word in words if word.lower() in methodology_keywords)
    issues['methodology_keyword_ratio'] = methods_count / len(words) if words else 0
    
    # 7. Check for result presentation
    result_indicators = {
        'table', 'figure', 'graph', 'plot', 'chart', 'diagram',
        'accuracy', 'precision', 'recall', 'f1', 'performance'
    }
    results_count = sum(1 for word in words if word.lower() in result_indicators)
    issues['has_results_presentation'] = results_count > 0
    
    # 8. Grammar and style checks
    consecutive_puncts = len(re.findall(r'[!?]{2,}', text))  # Multiple ! or ?
    all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))  # Words in ALL CAPS
    issues['style_issues_count'] = consecutive_puncts + all_caps_words
    
    # 9. Paragraph structure
    paragraphs = text.split('\n\n')
    short_paragraphs = sum(1 for p in paragraphs if len(p.split()) < 20)
    issues['short_paragraph_ratio'] = short_paragraphs / len(paragraphs) if paragraphs else 0
    
    return issues

def extract_text_features(text: str) -> Dict:
    """
    Extract features from cleaned text including quality metrics.
    
    Args:
        text (str): Cleaned text to analyze
        
    Returns:
        Dict: Dictionary containing extracted features
    """
    # Get basic features
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
    
    # Get quality issues
    quality_issues = detect_quality_issues(text)
    features.update(quality_issues)
    
    # Section presence detection (case-insensitive)
    section_patterns = {
        'has_methodology': r'\b(methodology|methods|experimental setup)\b',
        'has_results': r'\b(results|findings|observations)\b',
        'has_conclusion': r'\b(conclusion|conclusions|summary|final remarks)\b',
        'has_abstract': r'\b(abstract)\b',
        'has_introduction': r'\b(introduction)\b',
        'has_discussion': r'\b(discussion)\b'
    }
    
    # Check for presence of each section
    for feature_name, pattern in section_patterns.items():
        features[feature_name] = bool(re.search(pattern, text, re.IGNORECASE))
    
    # Additional features
    features['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
    
    # Fix: Count only words longer than 6 characters
    long_words = [word for word in words if len(word) > 6]
    features['long_words_ratio'] = len(long_words) / len(words) if words else 0
    
    # Technical content density
    technical_words = set([
        'algorithm', 'analysis', 'approach', 'data', 'method',
        'model', 'parameter', 'process', 'research', 'result',
        'study', 'system', 'technique', 'theory', 'variable'
    ])
    tech_word_count = sum(1 for word in words if word.lower() in technical_words)
    features['technical_density'] = tech_word_count / len(words) if words else 0
    
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

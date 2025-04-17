import pytest
from src.feature_engineering import extract_text_features, get_section_text, calculate_section_features

def test_extract_text_features_basic():
    """Test basic feature extraction with simple text"""
    text = "This is a test sentence. This is another sentence with some numbers 123."
    features = extract_text_features(text)
    
    assert features['word_count'] > 0
    assert features['sentence_count'] == 2
    assert features['avg_sentence_length'] > 0
    assert 'flesch_reading_ease' in features
    assert 'flesch_kincaid_grade' in features
    assert 'gunning_fog_index' in features

def test_extract_text_features_empty():
    """Test feature extraction with empty text"""
    features = extract_text_features("")
    
    assert features['word_count'] == 0
    assert features['sentence_count'] == 0
    assert features['avg_sentence_length'] == 0
    assert features['avg_word_length'] == 0
    assert features['long_words_ratio'] == 0

def test_extract_text_features_sections():
    """Test section detection"""
    text = """
    Abstract
    This is the abstract.
    
    Methodology
    This is the methodology section.
    
    Results
    These are the results.
    
    Conclusion
    This is the conclusion.
    """
    
    features = extract_text_features(text)
    assert features['has_methodology'] == True
    assert features['has_results'] == True
    assert features['has_conclusion'] == True

def test_get_section_text():
    """Test section text extraction"""
    text = """
    Abstract
    This is the abstract.
    
    Introduction
    This is the introduction.
    
    Methodology
    This is the methodology.
    
    Results
    These are the results.
    """
    
    abstract = get_section_text(text, "abstract")
    assert "This is the abstract" in abstract
    
    methodology = get_section_text(text, "methodology")
    assert "This is the methodology" in methodology

def test_get_section_text_missing():
    """Test handling of missing sections"""
    text = "This is some text without any sections."
    assert get_section_text(text, "abstract") == ""

def test_get_section_text_invalid():
    """Test handling of invalid section names"""
    text = "Some text"
    with pytest.raises(ValueError):
        get_section_text(text, "invalid_section")

def test_calculate_section_features():
    """Test feature calculation for all sections"""
    text = """
    Abstract
    This is the abstract.
    
    Introduction
    This is the introduction.
    
    Methodology
    This is the methodology.
    
    Results
    These are the results.
    
    Discussion
    This is the discussion.
    
    Conclusion
    This is the conclusion.
    """
    
    section_features = calculate_section_features(text)
    
    # Check if all sections are processed
    assert 'abstract_features' in section_features
    assert 'introduction_features' in section_features
    assert 'methodology_features' in section_features
    assert 'results_features' in section_features
    assert 'conclusion_features' in section_features
    
    # Check if features are calculated for each section
    for section in section_features.values():
        assert 'word_count' in section
        assert 'sentence_count' in section
        assert 'avg_sentence_length' in section

def test_calculate_section_features_empty():
    """Test section feature calculation with empty text"""
    section_features = calculate_section_features("")
    assert len(section_features) == 0

def test_readability_scores():
    """Test if readability scores are within expected ranges"""
    text = "This is a simple test sentence. It contains basic words. " * 5
    features = extract_text_features(text)
    
    assert 0 <= features['flesch_reading_ease'] <= 100
    assert features['flesch_kincaid_grade'] >= 0
    assert features['gunning_fog_index'] >= 0

def test_long_words_ratio():
    """Test calculation of long words ratio"""
    text = "short word verylongword another longerword"
    features = extract_text_features(text)
    
    # 3 long words (verylongword, another, longerword) out of 5 total words
    assert features['long_words_ratio'] == 0.6  # 3 out of 5 words are long (>6 characters) 
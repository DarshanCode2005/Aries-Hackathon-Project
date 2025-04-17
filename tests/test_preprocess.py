import pytest
import os
from src.preprocess import clean_text, extract_clean_text, preprocess_paper
from unittest.mock import patch, MagicMock

def test_clean_text_basic():
    """Test basic text cleaning without stopword removal"""
    test_text = "Hello World! 123 @#$%"
    expected = "hello world"
    assert clean_text(test_text) == expected

def test_clean_text_with_stopwords():
    """Test text cleaning with stopword removal"""
    test_text = "The quick brown fox jumps over the lazy dog"
    expected = "quick brown fox jumps lazy dog"
    assert clean_text(test_text, remove_stopwords=True) == expected

def test_clean_text_empty():
    """Test cleaning empty text"""
    assert clean_text("") == ""
    assert clean_text("", remove_stopwords=True) == ""

def test_clean_text_numbers_only():
    """Test cleaning text with only numbers"""
    assert clean_text("123 456 789") == ""

def test_clean_text_special_chars():
    """Test cleaning text with special characters"""
    test_text = "Hello! @World# $123%"
    expected = "hello world"
    assert clean_text(test_text) == expected

@pytest.fixture
def sample_pdf():
    """Create a temporary PDF file for testing"""
    # This is a mock test - in real scenario, you would create a temporary PDF file
    # For now, we'll just test if the function handles non-existent files correctly
    return "nonexistent.pdf"

def test_extract_clean_text_file_not_found(sample_pdf):
    """Test PDF extraction with non-existent file"""
    with pytest.raises(Exception):
        extract_clean_text(sample_pdf)

# Note: To properly test extract_clean_text, you would need:
# 1. A sample PDF file in your test directory
# 2. Additional test cases for header/footer removal
# 3. Test cases for references section removal
# Here's how you would write such tests:

"""
def test_extract_clean_text_with_sample_pdf():
    # Assuming you have a sample PDF file named 'sample_paper.pdf' in tests/data/
    pdf_path = os.path.join('tests', 'data', 'sample_paper.pdf')
    text = extract_clean_text(pdf_path)
    
    # Test if headers are removed
    assert "Running Header" not in text
    
    # Test if page numbers are removed
    assert not any(str(i) for i in range(1, 100) if str(i) in text.split())
    
    # Test if references section is removed
    assert "References" not in text
    assert "Bibliography" not in text
""" 

@pytest.fixture
def mock_pdf_text():
    return """
    Title of the Paper: Advanced Research Methods in Data Science
    
    Abstract
    This comprehensive research paper investigates advanced methodologies applied machine learning 
    techniques natural language processing deep learning algorithms. Research demonstrates significant 
    improvements performance accuracy compared traditional methods. Multiple experiments conducted 
    various datasets show consistent positive results across different domains applications.
    
    1. Introduction
    Machine learning algorithms continue revolutionize field artificial intelligence data science. 
    Recent developments neural networks transformer models demonstrate remarkable capabilities 
    processing analyzing complex data patterns. Research community actively explores novel approaches 
    improving existing methodologies developing innovative solutions challenging problems.
    
    2. Methodology
    Experimental setup included multiple processing stages data preparation feature extraction model 
    training evaluation. Implementation utilized state-of-art techniques ensuring robust reliable 
    results. Careful consideration given parameter selection optimization strategies validation 
    methods. Quality control measures implemented throughout process ensure reproducibility findings.
    
    3. Results
    Analysis revealed significant performance improvements compared baseline methods. Accuracy metrics 
    showed consistent enhancement across multiple test scenarios. Statistical validation confirmed 
    reliability findings confidence intervals demonstrated strong statistical significance. 
    Practical implications findings discussed context real-world applications.
    
    References
    [1] Smith, J. (2020) Advanced Machine Learning Methods
    [2] Johnson, M. (2021) Neural Network Applications
    """

def test_clean_text():
    text = "This is a TEST text with Numbers 123 and Symbols @#$%!"
    cleaned = clean_text(text)
    
    # Check lowercase
    assert cleaned.lower() == cleaned
    # Check special characters removed
    assert all(c.isalnum() or c.isspace() or c == '.' for c in cleaned)
    # Check extra whitespace removed
    assert "  " not in cleaned

@patch('pdfplumber.open')
def test_preprocess_paper_success(mock_pdf_open, tmp_path, mock_pdf_text):
    # Create mock PDF
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = mock_pdf_text
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf_open.return_value = mock_pdf
    
    # Create dummy PDF file
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()
    
    # Process the paper with a lower threshold for testing
    result = preprocess_paper(str(pdf_path), min_word_count=20)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert result.lower() == result  # Check lowercase
    assert "page 1" not in result.lower()  # Check page numbers removed
    assert "page 2" not in result.lower()
    
    # Additional checks for content
    assert "machine learning" in result.lower()
    assert "methodology" in result.lower()
    assert "results" in result.lower()
    assert "references" not in result.lower()  # References section should be removed

@patch('pdfplumber.open')
def test_preprocess_paper_empty_text(mock_pdf_open, tmp_path):
    # Mock PDF with no text
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf_open.return_value = mock_pdf
    
    # Create dummy PDF file
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.touch()
    
    result = preprocess_paper(str(pdf_path))
    assert result is None

@patch('pdfplumber.open')
def test_preprocess_paper_short_text(mock_pdf_open, tmp_path):
    # Mock PDF with very short text
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Short text"
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf_open.return_value = mock_pdf
    
    # Create dummy PDF file
    pdf_path = tmp_path / "short.pdf"
    pdf_path.touch()
    
    result = preprocess_paper(str(pdf_path))
    assert result is None

def test_preprocess_paper_invalid_file():
    result = preprocess_paper("nonexistent.pdf")
    assert result is None 
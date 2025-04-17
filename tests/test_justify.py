import pytest
from unittest.mock import patch, MagicMock
from src.justify import JustificationGenerator

@pytest.fixture
def sample_text():
    return """
    This paper presents a novel methodology for analyzing scientific data.
    Our experimental results show a significant improvement (p < 0.05) over baseline methods.
    The study is well-structured and follows a logical flow. We cite relevant work [1] and (Smith, 2020).
    """

@pytest.fixture
def justifier():
    with patch('transformers.pipeline') as mock_pipeline:
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_pipeline.return_value = MagicMock()
            mock_tokenizer.return_value = MagicMock()
            yield JustificationGenerator()

def test_extract_features(justifier, sample_text):
    features = justifier._extract_features(sample_text)
    
    assert isinstance(features, dict)
    assert all(k in features for k in ['methodology_score', 'evidence_score', 'coherence_score'])
    assert features['has_numbers'] == True
    assert features['has_citations'] == True
    assert 0 <= features['methodology_score'] <= 1
    assert 0 <= features['evidence_score'] <= 1
    assert 0 <= features['coherence_score'] <= 1

def test_generate_heuristic_justification(justifier, sample_text):
    features = justifier._extract_features(sample_text)
    justification = justifier._generate_heuristic_justification(features, prediction=1)
    
    assert isinstance(justification, str)
    assert len(justification.split()) <= 100
    assert "Classification: " in justification
    assert "methodology" in justification.lower()
    assert "evidence" in justification.lower()
    assert "citations present" in justification.lower()

def test_generate_model_summary(justifier, sample_text):
    mock_summary = "Generated summary"
    mock_summarizer = MagicMock()
    mock_summarizer.return_value = [{'summary_text': mock_summary}]
    justifier.summarizer = mock_summarizer
    
    summary = justifier._generate_model_summary(sample_text)
    assert isinstance(summary, str)
    assert summary == mock_summary
    mock_summarizer.assert_called_once()

def test_generate_justification_heuristic(justifier, sample_text):
    justification = justifier.generate_justification(sample_text, prediction=1, method='heuristic')
    
    assert isinstance(justification, str)
    assert len(justification.split()) <= 100
    assert "Classification: " in justification

def test_generate_justification_model(justifier, sample_text):
    mock_summary = "Model-generated justification"
    mock_summarizer = MagicMock()
    mock_summarizer.return_value = [{'summary_text': mock_summary}]
    justifier.summarizer = mock_summarizer
    
    justification = justifier.generate_justification(sample_text, prediction=1, method='model')
    assert isinstance(justification, str)
    assert justification == mock_summary
    mock_summarizer.assert_called_once()

def test_generate_justification_invalid_method(justifier, sample_text):
    justification = justifier.generate_justification(sample_text, prediction=1, method='invalid')
    assert "Unable to generate justification" in justification

def test_model_summary_error_handling(justifier, sample_text):
    mock_summarizer = MagicMock()
    mock_summarizer.side_effect = Exception("Model error")
    justifier.summarizer = mock_summarizer
    
    summary = justifier._generate_model_summary(sample_text)
    assert "Unable to generate summary" in summary
    mock_summarizer.assert_called_once()

def test_model_summary_empty_result(justifier, sample_text):
    mock_summarizer = MagicMock()
    mock_summarizer.return_value = []  # Empty result
    justifier.summarizer = mock_summarizer
    
    summary = justifier._generate_model_summary(sample_text)
    assert summary == "No summary generated."
    mock_summarizer.assert_called_once()

def test_model_summary_invalid_result(justifier, sample_text):
    mock_summarizer = MagicMock()
    mock_summarizer.return_value = [{}]  # Missing summary_text
    justifier.summarizer = mock_summarizer
    
    summary = justifier._generate_model_summary(sample_text)
    assert summary == "No summary generated."
    mock_summarizer.assert_called_once() 
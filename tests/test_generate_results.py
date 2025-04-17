import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from pathlib import Path
from src.generate_results import load_papers, generate_results

@pytest.fixture
def mock_pdf_dir(tmp_path):
    # Create test PDF files
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    
    # Create dummy PDF files
    (pdf_dir / "paper1.pdf").touch()
    (pdf_dir / "paper2.pdf").touch()
    
    return str(pdf_dir)

def test_load_papers(mock_pdf_dir):
    papers = load_papers(mock_pdf_dir)
    
    assert len(papers) == 2
    assert all(isinstance(p, tuple) and len(p) == 2 for p in papers)
    assert all(p[0] in ['paper1', 'paper2'] for p in papers)
    assert all(p[1].endswith('.pdf') for p in papers)

def test_load_papers_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    papers = load_papers(str(empty_dir))
    assert len(papers) == 0

@patch('src.generate_results.preprocess_paper')
@patch('src.generate_results.extract_features')
@patch('src.generate_results.PaperClassifier')
@patch('src.generate_results.JustificationGenerator')
def test_generate_results(mock_justifier_class, mock_classifier_class, 
                         mock_extract_features, mock_preprocess, mock_pdf_dir):
    # Set up mocks
    mock_classifier = MagicMock()
    mock_classifier_class.return_value = mock_classifier
    mock_classifier.predict.return_value = [1]  # Publishable
    mock_classifier.predict_proba.return_value = [[0.3, 0.7]]  # 70% confidence
    
    mock_justifier = MagicMock()
    mock_justifier_class.return_value = mock_justifier
    mock_justifier.generate_justification.return_value = "Test justification"
    
    mock_preprocess.return_value = "Preprocessed text"
    mock_extract_features.return_value = {
        'word_count': 1000,
        'citation_count': 15,
        'technical_score': 0.8,
        'readability_score': 0.7
    }
    
    # Run generate_results
    output_file = str(Path(mock_pdf_dir) / "results.csv")
    df = generate_results(
        data_dir=mock_pdf_dir,
        model_dir="mock_model_dir",
        output_file=output_file
    )
    
    # Verify results
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Two papers
    assert list(df.columns) == [
        'Paper_ID', 'Publishable', 'Justification', 'Confidence_Score',
        'Word_Count', 'Citation_Count', 'Technical_Score', 'Readability_Score'
    ]
    assert all(df['Publishable'] == 'Yes')
    assert all(df['Justification'] == 'Test justification')
    assert all(df['Confidence_Score'] == 0.7)
    
    # Verify file was saved
    assert os.path.exists(output_file)
    saved_df = pd.read_csv(output_file)
    assert len(saved_df) == len(df)

@patch('src.generate_results.preprocess_paper')
def test_generate_results_with_errors(mock_preprocess, mock_pdf_dir):
    # Make preprocess_paper fail for one paper
    def mock_preprocess_with_error(file_path):
        if 'paper1' in file_path:
            raise Exception("Test error")
        return "Preprocessed text"
    
    mock_preprocess.side_effect = mock_preprocess_with_error
    
    # Run generate_results
    output_file = str(Path(mock_pdf_dir) / "results.csv")
    df = generate_results(
        data_dir=mock_pdf_dir,
        model_dir="mock_model_dir",
        output_file=output_file
    )
    
    # Verify that only one paper was processed successfully
    assert len(df) == 1
    assert df.iloc[0]['Paper_ID'] == 'paper2'

def test_generate_results_no_papers(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    with pytest.raises(ValueError, match="No papers found"):
        generate_results(data_dir=str(empty_dir)) 
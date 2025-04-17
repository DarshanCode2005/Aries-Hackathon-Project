import pytest
import os
import pandas as pd
import shutil
from src.preprocess import process_earc_dataset, save_processed_data

@pytest.fixture
def test_data_directory(tmp_path):
    """Create a temporary directory structure with test PDF files"""
    # Create directory structure
    base_dir = tmp_path / "EARC_Dataset" / "Reference"
    pub_dir = base_dir / "Publishable"
    non_pub_dir = base_dir / "Non-Publishable"
    
    pub_dir.mkdir(parents=True)
    non_pub_dir.mkdir(parents=True)
    
    # Create a simple but valid PDF content with text
    sample_text = """
    Abstract
    This is a test paper abstract.
    
    Introduction
    This is the introduction section.
    
    Methodology
    This describes the methodology.
    
    Results
    These are the results.
    
    Conclusion
    This is the conclusion.
    """
    
    # Write sample text files (we'll mock PDF processing in the test)
    (pub_dir / "paper1.pdf").write_text(sample_text)
    (pub_dir / "paper2.pdf").write_text(sample_text)
    (non_pub_dir / "paper3.pdf").write_text(sample_text)
    (non_pub_dir / "paper4.pdf").write_text(sample_text)
    
    # Create a non-PDF file to test filtering
    (pub_dir / "not_a_paper.txt").write_text("This should be ignored")
    
    return str(base_dir)

def test_process_earc_dataset(test_data_directory, monkeypatch):
    """Test processing of the EARC dataset"""
    # Mock the extract_clean_text function to return predictable text
    def mock_extract_clean_text(pdf_path):
        return "This is test content from " + os.path.basename(pdf_path)
    
    # Apply the mock
    monkeypatch.setattr("src.preprocess.extract_clean_text", mock_extract_clean_text)
    
    # Process the dataset
    df = process_earc_dataset(test_data_directory)
    
    # Check basic DataFrame properties
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {'Paper_ID', 'Cleaned_Text', 'Label'}
    
    # Check number of papers processed
    assert len(df) == 4  # Should have processed 4 PDF files
    
    # Check labels
    assert len(df[df['Label'] == 'Publishable']) == 2
    assert len(df[df['Label'] == 'Non-Publishable']) == 2
    
    # Check Paper_IDs
    expected_ids = {'paper1', 'paper2', 'paper3', 'paper4'}
    assert set(df['Paper_ID']) == expected_ids
    
    # Check if text was extracted
    assert all(df['Cleaned_Text'].str.contains('This is test content'))

def test_process_earc_dataset_missing_directory():
    """Test handling of missing directory"""
    df = process_earc_dataset("nonexistent_directory")
    
    # Should return empty DataFrame with correct columns
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ['Paper_ID', 'Cleaned_Text', 'Label']

@pytest.fixture
def test_output_directory(tmp_path):
    """Create a temporary directory for output files"""
    output_dir = tmp_path / "processed"
    return str(output_dir)

def test_save_processed_data(test_output_directory):
    """Test saving processed data to files"""
    # Create test DataFrame
    test_data = {
        'Paper_ID': ['paper1', 'paper2'],
        'Cleaned_Text': ['text1', 'text2'],
        'Label': ['Publishable', 'Non-Publishable']
    }
    df = pd.DataFrame(test_data)
    
    # Save the data
    save_processed_data(df, test_output_directory)
    
    # Check if files were created
    assert os.path.exists(os.path.join(test_output_directory, 'processed_papers.csv'))
    assert os.path.exists(os.path.join(test_output_directory, 'processed_papers.pkl'))
    
    # Load and verify CSV
    loaded_csv = pd.read_csv(os.path.join(test_output_directory, 'processed_papers.csv'))
    assert len(loaded_csv) == len(df)
    assert all(loaded_csv.columns == df.columns)
    
    # Load and verify pickle
    loaded_pkl = pd.read_pickle(os.path.join(test_output_directory, 'processed_papers.pkl'))
    assert len(loaded_pkl) == len(df)
    assert all(loaded_pkl.columns == df.columns)

def test_save_processed_data_empty_df(test_output_directory):
    """Test saving an empty DataFrame"""
    df = pd.DataFrame(columns=['Paper_ID', 'Cleaned_Text', 'Label'])
    save_processed_data(df, test_output_directory)
    
    # Check if files were created
    assert os.path.exists(os.path.join(test_output_directory, 'processed_papers.csv'))
    assert os.path.exists(os.path.join(test_output_directory, 'processed_papers.pkl')) 
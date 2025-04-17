import pytest
import numpy as np
import pandas as pd
import os
import json
from unittest.mock import Mock, patch, MagicMock
from src.model_train import PaperClassifier, train_and_evaluate
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    texts = [
        "This is a publishable paper with good methodology.",
        "This paper lacks proper methodology and results.",
        "Excellent research with clear conclusions.",
        "Poor research design and incomplete results."
    ]
    labels = ['Publishable', 'Non-Publishable', 'Publishable', 'Non-Publishable']
    return texts, labels

@pytest.fixture
def mock_embeddings():
    """Create mock embeddings"""
    return np.random.rand(4, 384)  # 384 is the dimension for 'all-MiniLM-L6-v2'

@pytest.fixture
def classifier():
    """Create a PaperClassifier instance"""
    return PaperClassifier()

def test_classifier_initialization(classifier):
    """Test if PaperClassifier initializes correctly"""
    assert hasattr(classifier, 'embedding_model')
    assert hasattr(classifier, 'classifier')
    assert hasattr(classifier, 'label_encoder')
    assert isinstance(classifier.embedding_model, SentenceTransformer)
    assert isinstance(classifier.classifier, LogisticRegression)
    assert isinstance(classifier.label_encoder, LabelEncoder)

@patch('src.model_train.SentenceTransformer')
def test_generate_embeddings(mock_st, classifier, sample_data):
    """Test embedding generation"""
    texts, _ = sample_data
    mock_embeddings = np.random.rand(len(texts), 384)
    
    # Create a new mock for the encoder
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = mock_embeddings
    
    # Replace the real encoder with our mock
    classifier.embedding_model = mock_encoder
    
    embeddings = classifier.generate_embeddings(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(texts), 384)
    mock_encoder.encode.assert_called_once_with(texts, show_progress_bar=True)

def test_train_and_predict(classifier, sample_data, mock_embeddings):
    texts, labels = sample_data
    
    # Train the model
    encoded_labels = classifier.label_encoder.fit_transform(labels)
    classifier.train(mock_embeddings, encoded_labels)
    assert classifier.classifier is not None
    
    # Test predictions
    predictions = classifier.predict(mock_embeddings)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(texts)
    assert all(isinstance(pred, (np.int64, int)) for pred in predictions)

def test_evaluate(classifier, sample_data, mock_embeddings):
    texts, labels = sample_data
    
    encoded_labels = classifier.label_encoder.fit_transform(labels)
    classifier.train(mock_embeddings, encoded_labels)
    metrics = classifier.evaluate(mock_embeddings, encoded_labels)
    
    assert isinstance(metrics, dict)
    assert all(metric in metrics for metric in ['accuracy', 'f1_score', 'classification_report'])
    assert isinstance(metrics['accuracy'], float)
    assert isinstance(metrics['f1_score'], float)
    assert isinstance(metrics['classification_report'], str)

def test_save_and_load_model(classifier, sample_data, mock_embeddings, tmp_path):
    texts, labels = sample_data
    model_path = tmp_path / "models"
    
    # Train and save the model
    encoded_labels = classifier.label_encoder.fit_transform(labels)
    classifier.train(mock_embeddings, encoded_labels)
    classifier.save_model(model_path)
    
    # Verify files were created
    assert os.path.exists(model_path)
    assert os.path.exists(model_path / "classifier.joblib")
    assert os.path.exists(model_path / "label_encoder.joblib")
    
    # Load the model in a new classifier
    new_classifier = PaperClassifier()
    new_classifier.load_model(model_path)
    
    # Verify predictions match
    original_preds = classifier.predict(mock_embeddings)
    loaded_preds = new_classifier.predict(mock_embeddings)
    np.testing.assert_array_equal(original_preds, loaded_preds)

@patch('src.model_train.PaperClassifier')
def test_train_and_evaluate_function(mock_classifier_class, tmp_path):
    """Test the complete training and evaluation pipeline"""
    mock_classifier = MagicMock()
    mock_classifier_class.return_value = mock_classifier
    
    # Mock the generate_embeddings method to return a numpy array
    mock_embeddings = np.random.rand(10, 384)  # 10 samples
    mock_classifier.generate_embeddings.return_value = mock_embeddings
    
    # Mock the label_encoder's fit_transform to return numpy array
    mock_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 10 samples, balanced classes
    mock_classifier.label_encoder.fit_transform.return_value = mock_labels
    
    # Mock the evaluate method
    mock_classifier.evaluate.return_value = {
        'accuracy': 0.9,
        'f1_score': 0.85,
        'classification_report': 'Mock Report'
    }
    
    data_path = tmp_path / "data.pkl"
    model_path = tmp_path / "models"
    
    # Create sample data file with multiple samples
    df = pd.DataFrame({
        'Cleaned_Text': [f"sample text {i+1}" for i in range(10)],
        'Label': ["Non-Publishable" if i % 2 == 0 else "Publishable" for i in range(10)]
    })
    df.to_pickle(data_path)
    
    metrics = train_and_evaluate(str(data_path), str(model_path))
    
    # Verify the mock methods were called
    mock_classifier.generate_embeddings.assert_called_once()
    mock_classifier.label_encoder.fit_transform.assert_called_once()
    mock_classifier.train.assert_called_once()
    mock_classifier.evaluate.assert_called_once()
    mock_classifier.save_model.assert_called_once()
    
    # Verify the results
    assert isinstance(metrics, dict)
    assert 'metrics' in metrics
    assert 'classification_report' in metrics
    assert metrics['metrics']['accuracy'] == 0.9
    assert metrics['metrics']['f1_score'] == 0.85

def test_train_and_evaluate_invalid_path():
    """Test handling of invalid data path"""
    with pytest.raises(FileNotFoundError):
        train_and_evaluate("invalid_path.json", "models") 
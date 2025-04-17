import pytest
import numpy as np
from src.model_train import PaperClassifier

@pytest.fixture
def sample_data():
    texts = [
        "This is a machine learning paper",
        "Deep learning in computer vision",
        "Statistical analysis of data",
        "Natural language processing advances"
    ]
    labels = [1, 1, 0, 1]
    return texts, labels

@pytest.fixture
def mock_embeddings(sample_data):
    texts, _ = sample_data
    return np.random.rand(len(texts), 384)  # 384 is the default embedding dimension

@pytest.fixture
def classifier():
    return PaperClassifier() 
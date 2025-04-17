import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import logging
from typing import Tuple, Dict
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperClassifier:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the paper classifier with specified embedding model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(model_name)
        # Added L2 regularization and balanced class weights
        self.classifier = LogisticRegression(
            max_iter=1000,
            C=1.0,  # L2 regularization strength
            class_weight='balanced',
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        
    def generate_embeddings(self, texts: list) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (list): List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def train(self, X_train: np.ndarray, y_train: np.array) -> None:
        """
        Train the logistic regression classifier.
        
        Args:
            X_train (np.ndarray): Training features (embeddings)
            y_train (np.array): Training labels
        """
        logger.info("Training classifier...")
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Update classifier with computed weights
        self.classifier.set_params(class_weight=class_weight_dict)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        self.classifier.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.array:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            np.array: Predicted labels
        """
        return self.classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        return self.classifier.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.array) -> Dict:
        """
        Evaluate the model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.array): True labels
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred),
            'predictions': {
                'true_labels': y_test.tolist(),
                'predicted_labels': y_pred.tolist()
            }
        }
        
        return metrics
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model and label encoder.
        
        Args:
            output_dir (str): Directory to save model files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the classifier
        joblib.dump(self.classifier, os.path.join(output_dir, 'classifier.joblib'))
        
        # Save the label encoder
        joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
    
    def load_model(self, model_dir: str) -> None:
        """
        Load a trained model and label encoder.
        
        Args:
            model_dir (str): Directory containing model files
        """
        self.classifier = joblib.load(os.path.join(model_dir, 'classifier.joblib'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))

def train_and_evaluate(data_path: str, output_dir: str = 'models') -> Dict:
    """
    Train and evaluate the paper classifier on the dataset.
    
    Args:
        data_path (str): Path to the processed papers DataFrame
        output_dir (str): Directory to save model and results
        
    Returns:
        Dict: Dictionary containing training results and metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    df = pd.read_pickle(data_path)
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Validate data
    required_columns = ['Cleaned_Text', 'Label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check class balance
    class_counts = df['Label'].value_counts()
    logger.info(f"Class distribution:\n{class_counts}")
    
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes for training")
    
    min_class_size = class_counts.min()
    if min_class_size < 3:
        raise ValueError(f"Insufficient samples in smallest class: {min_class_size}")
    
    # Initialize classifier
    classifier = PaperClassifier()
    
    # Generate embeddings
    embeddings = classifier.generate_embeddings(df['Cleaned_Text'].tolist())
    
    # Encode labels
    labels = classifier.label_encoder.fit_transform(df['Label'])
    
    # Split data
    if len(labels) < 10:
        logger.warning("Small dataset detected, using larger test split")
        test_size = 0.4
    else:
        test_size = 0.2
        
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    # Train model
    logger.info(f"Training model with {len(X_train)} samples...")
    classifier.train(X_train, y_train)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    
    # Validate model performance
    if metrics['accuracy'] < 0.5:
        logger.warning("Model performance is poor (accuracy < 0.5)")
    
    # Save model
    classifier.save_model(output_dir)
    
    # Save metrics
    results = {
        'metrics': {
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score']
        },
        'classification_report': metrics['classification_report'],
        'training_params': {
            'test_size': test_size,
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test),
            'class_distribution': class_counts.to_dict()
        }
    }
    
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Training accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Training F1 score: {metrics['f1_score']:.4f}")
    logger.info("\nClassification Report:\n" + metrics['classification_report'])
    
    return results

if __name__ == "__main__":
    # Train and evaluate model
    results = train_and_evaluate('data/processed/processed_papers.pkl')
    
    # Print results
    print("\nTraining Results:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])

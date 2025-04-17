import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC  # Adding SVM as an alternative classifier
import joblib
import os
import logging
from typing import Tuple, Dict
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperClassifier:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', n_splits: int = 5):
        """
        Initialize the paper classifier with specified embedding model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            n_splits (int): Number of cross-validation splits
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.cv_results = None
        self.n_splits = n_splits
        
    def generate_embeddings(self, texts: list) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (list): List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings
            
        Raises:
            ValueError: If texts is empty or None
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for empty text list")
            
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def get_cv_scores(self, X: np.ndarray, y: np.array) -> Dict:
        """
        Get cross-validation scores.
        
        Args:
            X (np.ndarray): Features
            y (np.array): Labels
            
        Returns:
            Dict: Cross-validation scores
        """
        if self.cv_results is None:
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='f1_weighted')
            self.cv_results = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std() * 2,
                'individual_scores': cv_scores.tolist()
            }
        return self.cv_results
    
    def train(self, X_train: np.ndarray, y_train: np.array) -> None:
        """
        Train the classifier with hyperparameter tuning.
        
        Args:
            X_train (np.ndarray): Training features (embeddings)
            y_train (np.array): Training labels
        """
        logger.info("Training classifier with hyperparameter tuning...")
        
        # Always fit the label encoder, regardless of input type
        self.label_encoder.fit(y_train)
        y_train_encoded = self.label_encoder.transform(y_train)
        
        # Get the size of the smallest class
        min_class_size = np.min(np.bincount(y_train_encoded))
        
        # Adjust n_splits based on the smallest class size
        n_splits = min(self.n_splits, min_class_size)
        logger.info(f"Using {n_splits}-fold cross-validation based on smallest class size of {min_class_size}")
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_encoded),
            y=y_train_encoded
        )
        class_weight_dict = dict(zip(np.unique(y_train_encoded), class_weights))
        
        # Try multiple classifiers including Random Forest
        pipelines = {
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    random_state=42,
                    n_jobs=-1,  # Use all CPU cores
                    class_weight='balanced',
                    n_estimators=200  # Start with a reasonable number of trees
                ))
            ]),
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=42,
                    max_iter=5000,
                    class_weight='balanced'
                ))
            ])
        }
        
        # Simplified parameter grids to reduce computation time
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__max_features': ['sqrt']
            },
            'logistic': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l2']
            }
        }
        
        # Set up cross-validation with reduced splits
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        best_score = 0
        best_pipeline = None
        best_params = None
        all_scores = {}
        
        # Try each classifier
        for name, pipeline in pipelines.items():
            logger.info(f"\nTrying {name} classifier...")
            
            try:
                # Perform grid search
                grid_search = GridSearchCV(
                    pipeline,
                    param_grids[name],
                    cv=cv,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train_encoded)
                
                # Store scores for this classifier
                all_scores[name] = {
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_
                }
                
                logger.info(f"{name} best parameters: {grid_search.best_params_}")
                logger.info(f"{name} best score: {grid_search.best_score_:.4f}")
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_pipeline = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
            except Exception as e:
                logger.warning(f"Error during {name} classifier training: {str(e)}")
                continue
        
        if best_pipeline is None:
            raise ValueError("No classifier was successfully trained")
        
        # Log all classifier performances
        logger.info("\nAll classifier performances:")
        for name, score_info in all_scores.items():
            logger.info(f"{name}: {score_info['best_score']:.4f}")
        
        logger.info(f"\nBest overall model: {type(best_pipeline.named_steps['classifier']).__name__}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation score: {best_score:.4f}")
        
        # Update pipeline with best model
        self.pipeline = best_pipeline
        
        # Perform final cross-validation
        cv_scores = cross_val_score(self.pipeline, X_train, y_train_encoded, cv=cv, scoring='f1_weighted')
        logger.info(f"Final CV scores: {cv_scores}")
        logger.info(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store CV results
        self.cv_results = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std() * 2,
            'individual_scores': cv_scores.tolist(),
            'best_model_type': type(best_pipeline.named_steps['classifier']).__name__,
            'best_parameters': best_params,
            'all_model_scores': all_scores,
            'n_splits': n_splits
        }
        
    def predict(self, X: np.ndarray) -> np.array:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            np.array: Predicted labels as np.int64
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
            
        if not hasattr(self.label_encoder, 'classes_'):
            raise ValueError("Label encoder not fitted. Train the model first.")
            
        predictions = self.pipeline.predict(X)
        return np.array(predictions, dtype=np.int64)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.pipeline.predict_proba(X)
    
    def get_feature_importance(self, X: np.ndarray) -> Dict:
        """
        Get feature importance analysis.
        
        Args:
            X (np.ndarray): Features to analyze
            
        Returns:
            Dict: Feature importance metrics and analysis
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get feature importance based on model type
        classifier = self.pipeline.named_steps['classifier']
        importance_dict = {}
        
        if isinstance(classifier, RandomForestClassifier):
            # For Random Forest, get direct feature importance
            importance = classifier.feature_importances_
            importance_dict['feature_importance'] = importance.tolist()
            importance_dict['importance_type'] = 'feature_importances'
            
        elif isinstance(classifier, LogisticRegression):
            # For Logistic Regression, get coefficients
            importance = np.abs(classifier.coef_[0])  # Use absolute values
            importance_dict['feature_importance'] = importance.tolist()
            importance_dict['importance_type'] = 'coefficients'
            
        # Get top K most important dimensions
        K = 10  # Number of top features to analyze
        top_indices = np.argsort(importance)[-K:][::-1]
        
        importance_dict['top_features'] = {
            'indices': top_indices.tolist(),
            'values': importance[top_indices].tolist()
        }
        
        # Add model-specific analysis
        importance_dict['model_type'] = type(classifier).__name__
        if isinstance(classifier, LogisticRegression):
            importance_dict['regularization_strength'] = 1/classifier.C
            
        return importance_dict
        
    def evaluate(self, X_test: np.ndarray, y_test: np.array) -> Dict:
        """
        Evaluate the model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.array): True labels (can be strings or numbers)
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not hasattr(self.label_encoder, 'classes_'):
            raise ValueError("Label encoder not fitted. Train the model first.")
            
        # Get predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Transform test labels to encoded format
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Calculate metrics using encoded labels
        metrics = {
            'accuracy': accuracy_score(y_test_encoded, y_pred),
            'f1_score': f1_score(y_test_encoded, y_pred, average='weighted'),
            'classification_report': classification_report(y_test_encoded, y_pred),
            'predictions': {
                'true_labels': y_test.tolist(),  # Keep original labels in output
                'predicted_labels': self.label_encoder.inverse_transform(y_pred).tolist(),  # Convert back to original labels
                'probabilities': y_proba.tolist()
            }
        }
        
        # Add feature importance analysis
        metrics['feature_importance'] = self.get_feature_importance(X_test)
        
        # Add prediction analysis
        metrics['prediction_analysis'] = {
            'mean_probability_class_0': float(np.mean(y_proba[:, 0])),
            'mean_probability_class_1': float(np.mean(y_proba[:, 1])),
            'std_probability_class_0': float(np.std(y_proba[:, 0])),
            'std_probability_class_1': float(np.std(y_proba[:, 1])),
            'prediction_confidence': float(np.mean(np.max(y_proba, axis=1)))
        }
        
        return metrics
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model and components.
        
        Args:
            output_dir (str): Directory to save model files
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the pipeline
        joblib.dump(self.pipeline, os.path.join(output_dir, 'classifier.joblib'))
        
        # Save the label encoder
        joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))
        
        # Save cross-validation results if available
        if self.cv_results:
            with open(os.path.join(output_dir, 'cv_results.json'), 'w') as f:
                json.dump(self.cv_results, f)
    
    def load_model(self, model_dir: str) -> None:
        """
        Load a trained model and components.
        
        Args:
            model_dir (str): Directory containing model files
        """
        self.pipeline = joblib.load(os.path.join(model_dir, 'classifier.joblib'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
        
        # Load CV results if available
        cv_results_path = os.path.join(model_dir, 'cv_results.json')
        if os.path.exists(cv_results_path):
            with open(cv_results_path, 'r') as f:
                self.cv_results = json.load(f)

def train_and_evaluate(data_path: str, output_dir: str = 'models') -> Dict:
    """
    Train and evaluate the paper classifier with improved validation.
    
    Args:
        data_path (str): Path to the processed papers DataFrame
        output_dir (str): Directory to save model and results
        
    Returns:
        Dict: Dictionary containing training results and metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and validate data
    logger.info("Loading data...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    df = pd.read_pickle(data_path)
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Validate required columns
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
    classifier = PaperClassifier(n_splits=5)  # Increased from 3 to 5 for better validation
    
    # Generate embeddings
    logger.info("Generating embeddings for all papers...")
    embeddings = classifier.generate_embeddings(df['Cleaned_Text'].tolist())
    labels = df['Label'].values
    
    # Perform stratified split with larger test size
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, 
        labels,
        test_size=0.3,  # Increased test size for better evaluation
        stratify=labels,
        random_state=42
    )
    
    # Log split sizes
    logger.info(f"Training set size: {len(X_train)} ({np.unique(y_train, return_counts=True)[1].tolist()})")
    logger.info(f"Test set size: {len(X_test)} ({np.unique(y_test, return_counts=True)[1].tolist()})")
    
    # Train model
    logger.info("Training model...")
    classifier.train(X_train, y_train)
    
    # Get cross-validation results
    cv_results = classifier.get_cv_scores(X_train, y_train)
    logger.info("\nCross-validation Results:")
    logger.info(f"Individual CV scores: {cv_results['individual_scores']}")
    logger.info(f"Mean CV score: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score']:.4f})")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    metrics = classifier.evaluate(X_test, y_test)
    
    # Detailed evaluation logging
    logger.info(f"\nTest Set Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"\nClassification Report:\n{metrics['classification_report']}")
    
    # Calculate and log prediction probabilities distribution
    y_proba = classifier.predict_proba(X_test)
    class_probs = {
        'class_0_mean_prob': float(np.mean(y_proba[:, 0])),
        'class_1_mean_prob': float(np.mean(y_proba[:, 1])),
        'class_0_std_prob': float(np.std(y_proba[:, 0])),
        'class_1_std_prob': float(np.std(y_proba[:, 1]))
    }
    logger.info("\nPrediction Probability Distribution:")
    logger.info(f"Class 0 (Non-publishable) - Mean: {class_probs['class_0_mean_prob']:.4f}, Std: {class_probs['class_0_std_prob']:.4f}")
    logger.info(f"Class 1 (Publishable) - Mean: {class_probs['class_1_mean_prob']:.4f}, Std: {class_probs['class_1_std_prob']:.4f}")
    
    # Add probability metrics to results
    metrics['probability_metrics'] = class_probs
    metrics['cross_validation_results'] = cv_results
    
    # Save model and results
    classifier.save_model(output_dir)
    
    # Save detailed metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Best model type: {cv_results.get('best_model_type', 'Not available')}")
    logger.info(f"Best parameters: {cv_results.get('best_parameters', 'Not available')}")
    
    return metrics

if __name__ == "__main__":
    try:
        results = train_and_evaluate('data/processed/processed_papers.pkl')
        print(f"\nTraining Summary:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Cross-validation mean score: {results['cross_validation_results']['mean_cv_score']:.4f}")
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {str(e)}")
        raise

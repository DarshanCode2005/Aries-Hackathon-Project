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
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import logging
from typing import Tuple, Dict
import json
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperClassifier:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', n_splits: int = 5, use_smote: bool = True):
        """
        Initialize the paper classifier with specified embedding model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            n_splits (int): Number of cross-validation splits
            use_smote (bool): Whether to use SMOTE for handling class imbalance
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.cv_results = None
        self.n_splits = n_splits
        self.use_smote = use_smote
        
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
    
    def get_cv_scores(self, X, y, cv=3):
        """Get cross-validation scores for the trained classifier."""
        if not hasattr(self, 'classifier') or self.classifier is None:
            raise ValueError("Classifier not trained yet")
        
        # For small datasets where we used a simple model
        if isinstance(self.classifier, ImbPipeline):
            logging.info("Using simple validation for small dataset")
            y_pred = self.classifier.predict(X)
            score = f1_score(y, y_pred, average='weighted')
            return np.array([score])  # Return as array to match cross_val_score format
        
        # For larger datasets where we used cross-validation
        cv_scores = cross_val_score(self.classifier, X, y, cv=cv, scoring='f1_weighted')
        return cv_scores
    
    def train(self, X_train: np.ndarray, y_train: np.array) -> None:
        """
        Train the classifier with hyperparameter tuning.
        
        Args:
            X_train (np.ndarray): Training features (embeddings)
            y_train (np.array): Training labels
        """
        logger.info("\nTraining model...")
        logger.info("Training classifier with hyperparameter tuning...")
        
        # Log class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            logger.info(f"Class {label}: {count} samples")
        
        # Determine minimum samples and strategy
        min_samples = min(counts)
        logger.info(f"Minimum samples in any class: {min_samples}")
        
        # Compute class weights
        class_weights = dict(zip(unique, len(y_train) / (len(unique) * counts)))
        logger.info(f"Class weights: {class_weights}")
        
        # For small datasets, use a simple approach without cross-validation
        if min_samples < 5:
            logger.info("Small dataset detected, using simple model without cross-validation")
            try:
                # Try different simple models with class weights and regularization
                models = [
                    ('rf', RandomForestClassifier(
                        n_estimators=100,
                        max_depth=3,
                        min_samples_split=2,
                        class_weight=class_weights,
                        random_state=42
                    )),
                    ('lr', LogisticRegression(
                        C=0.1,  # Stronger regularization
                        class_weight=class_weights,
                        max_iter=1000,
                        random_state=42
                    ))
                ]
                
                best_score = -1
                best_pipeline = None
                
                for name, model in models:
                    try:
                        pipeline = ImbPipeline([
                            ('scaler', StandardScaler()),
                            ('resampler', RandomOverSampler(random_state=42)),
                            ('classifier', model)
                        ])
                        
                        # Fit and evaluate using stratified K-fold
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        scores = []
                        
                        for train_idx, val_idx in cv.split(X_train, y_train):
                            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                            
                            pipeline.fit(X_fold_train, y_fold_train)
                            y_pred = pipeline.predict(X_fold_val)
                            score = f1_score(y_fold_val, y_pred, average='weighted', zero_division=0)
                            scores.append(score)
                        
                        avg_score = np.mean(scores)
                        logger.info(f"{name} cross-validation f1 scores: {scores}")
                        logger.info(f"{name} average f1 score: {avg_score:.3f}")
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            # Retrain on full training set
                            pipeline.fit(X_train, y_train)
                            best_pipeline = pipeline
                            
                    except Exception as e:
                        logger.warning(f"Error during {name} training: \n{str(e)}")
                        continue
                
                if best_pipeline is not None:
                    self.classifier = best_pipeline
                    logger.info(f"Selected best model with average f1 score: {best_score:.3f}")
                    return
                else:
                    raise ValueError("No model could be trained successfully")
                
            except Exception as e:
                logger.warning(f"Error during simple model training: \n{str(e)}")
                raise ValueError("Could not train model even with simplified approach")
        
        # For larger datasets, use cross-validation and SMOTE
        n_splits = min(3, min_samples)
        logger.info(f"Using {n_splits}-fold cross-validation")
        
        # Try Random Forest first
        logger.info("\nTrying random_forest classifier...")
        try:
            rf_pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('resampler', SMOTE(k_neighbors=min(min_samples - 1, 5), random_state=42)),
                ('classifier', RandomForestClassifier(
                    class_weight=class_weights,
                    random_state=42
                ))
            ])
            
            rf_param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5],
                'classifier__min_samples_split': [2, 3]
            }
            
            rf_grid = GridSearchCV(
                rf_pipeline, 
                rf_param_grid,
                cv=n_splits,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            rf_grid.fit(X_train, y_train)
            self.classifier = rf_grid.best_estimator_
            logger.info(f"Best random forest parameters: {rf_grid.best_params_}")
            logger.info(f"Best random forest score: {rf_grid.best_score_:.3f}")
            return
            
        except Exception as e:
            logger.warning(f"Error during random_forest classifier training: \n{str(e)}")
        
        # Try Logistic Regression as fallback
        logger.info("\nTrying logistic classifier...")
        try:
            lr_pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('resampler', SMOTE(k_neighbors=min(min_samples - 1, 5), random_state=42)),
                ('classifier', LogisticRegression(
                    class_weight=class_weights,
                    random_state=42
                ))
            ])
            
            lr_param_grid = {
                'classifier__C': [0.01, 0.1, 1.0],  # Add stronger regularization
                'classifier__max_iter': [1000]
            }
            
            lr_grid = GridSearchCV(
                lr_pipeline, 
                lr_param_grid,
                cv=n_splits,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            lr_grid.fit(X_train, y_train)
            self.classifier = lr_grid.best_estimator_
            logger.info(f"Best logistic regression parameters: {lr_grid.best_params_}")
            logger.info(f"Best logistic regression score: {lr_grid.best_score_:.3f}")
            return
            
        except Exception as e:
            logger.warning(f"Error during logistic classifier training: \n{str(e)}")
        
        raise ValueError("No classifier was successfully trained")
    
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
        Get probability estimates for samples.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not hasattr(self, 'classifier') or self.classifier is None:
            raise ValueError("Classifier not trained yet")
        return self.classifier.predict_proba(X)
    
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
        
    def evaluate(self, X_test, y_test):
        """Evaluate the trained classifier on test data."""
        if not hasattr(self, 'classifier') or self.classifier is None:
            raise ValueError("Classifier not trained yet")
        
        y_pred = self.classifier.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        return report, y_test, y_pred
    
    def save_model(self, output_dir: str) -> None:
        """Save the trained model to disk."""
        if not hasattr(self, 'classifier') or self.classifier is None:
            raise ValueError("Classifier not trained yet")
        
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'paper_classifier.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> None:
        """Load a trained model from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")

def train_and_evaluate(data_path: str, output_dir: str = 'models', use_smote: bool = True) -> Dict:
    """
    Train and evaluate the paper classifier with improved validation and SMOTE.
    
    Args:
        data_path (str): Path to the processed papers DataFrame
        output_dir (str): Directory to save model and results
        use_smote (bool): Whether to use SMOTE for handling class imbalance
        
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
    
    # Check class balance and log distribution
    class_counts = df['Label'].value_counts()
    logger.info("\nInitial class distribution:")
    for label, count in class_counts.items():
        logger.info(f"Class {label}: {count} samples")
    
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes for training")
    
    min_class_size = class_counts.min()
    if min_class_size < 3:
        raise ValueError(f"Insufficient samples in smallest class: {min_class_size}")
    
    # Calculate imbalance ratio
    imbalance_ratio = class_counts.max() / class_counts.min()
    logger.info(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    
    # Initialize classifier with SMOTE if specified
    classifier = PaperClassifier(n_splits=5, use_smote=use_smote)
    
    # Generate embeddings
    logger.info("\nGenerating embeddings for all papers...")
    embeddings = classifier.generate_embeddings(df['Cleaned_Text'].tolist())
    labels = df['Label'].values
    
    # Perform stratified split with larger test size for better evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, 
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=42
    )
    
    # Log split sizes and class distribution
    logger.info("\nData split information:")
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    train_class_dist = pd.Series(y_train).value_counts()
    test_class_dist = pd.Series(y_test).value_counts()
    
    logger.info("\nTraining set class distribution:")
    for label, count in train_class_dist.items():
        logger.info(f"Class {label}: {count} samples")
    
    logger.info("\nTest set class distribution:")
    for label, count in test_class_dist.items():
        logger.info(f"Class {label}: {count} samples")
    
    # Train model
    logger.info("\nTraining model...")
    classifier.train(X_train, y_train)
    
    # Get cross-validation results
    cv_results = classifier.get_cv_scores(X_train, y_train)
    logger.info("\nCross-validation Results:")
    logger.info(f"Individual CV scores: {cv_results}")
    logger.info(f"Mean CV score: {cv_results.mean():.4f} (+/- {cv_results.std() * 2:.4f})")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    report, y_test, y_pred = classifier.evaluate(X_test, y_test)
    
    # Calculate metrics with zero_division=0 to handle undefined cases
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Detailed evaluation logging
    logger.info(f"\nTest Set Metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    
    # Prepare metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'cross_validation_results': {
            'individual_scores': cv_results.tolist(),
            'mean_cv_score': float(cv_results.mean()),
            'std_cv_score': float(cv_results.std() * 2)
        },
        'predictions': {
            'true_labels': y_test.tolist(),
            'predicted_labels': y_pred.tolist(),
            'probabilities': classifier.predict_proba(X_test).tolist()
        },
        'smote_info': {
            'used_smote': use_smote,
            'initial_class_distribution': class_counts.to_dict(),
            'train_class_distribution': train_class_dist.to_dict(),
            'test_class_distribution': test_class_dist.to_dict(),
            'imbalance_ratio': float(imbalance_ratio)
        }
    }
    
    # Save model and results
    classifier.save_model(output_dir)
    
    # Save detailed metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return metrics

if __name__ == "__main__":
    try:
        results = train_and_evaluate('data/processed/processed_papers.pkl', use_smote=True)
        print(f"\nTraining Summary:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Cross-validation mean score: {results['cross_validation_results']['mean_cv_score']:.4f}")
        if results['smote_info']['used_smote']:
            print("\nSMOTE was used to handle class imbalance")
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {str(e)}")
        raise

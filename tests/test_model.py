import unittest
import numpy as np
import pandas as pd
from src.model_train import PaperClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tempfile
import os

class TestPaperClassifier(unittest.TestCase):
    def setUp(self):
        """Set up test data with various scenarios."""
        # Create balanced sample data
        self.texts_balanced = [
            "This is a high-quality research paper with strong methodology and clear results.",
            "The results are well-supported by experimental data and statistical analysis.",
            "Excellent research design with comprehensive analysis and strong conclusions.",
            "Novel approach with rigorous validation and significant contributions.",
            "Clear methodology with reproducible results and thorough discussion.",
            "Poor methodology and lacks proper citations or experimental validation.",
            "Insufficient evidence and unclear conclusions with poor writing.",
            "No clear research contribution and lacks proper literature review.",
            "Methodology is flawed and results are not statistically significant.",
            "Poor structure with minimal analysis and inadequate discussion."
        ]
        # 5 publishable (1) and 5 non-publishable (0) papers
        self.labels_balanced = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        
        # Create imbalanced sample data
        self.texts_imbalanced = self.texts_balanced[:7]  # 5 positive, 2 negative
        self.labels_imbalanced = np.array([1, 1, 1, 1, 1, 0, 0])
        
        # Create small sample data
        self.texts_small = self.texts_balanced[:4]  # 2 positive, 2 negative
        self.labels_small = np.array([1, 1, 0, 0])
        
        # Initialize classifier
        self.classifier = PaperClassifier()
        
    def test_embedding_generation(self):
        """Test embedding generation for different dataset sizes."""
        # Test with balanced dataset
        embeddings = self.classifier.generate_embeddings(self.texts_balanced)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(len(embeddings), len(self.texts_balanced))
        self.assertTrue(embeddings.shape[1] > 0)
        
        # Test with small dataset
        embeddings_small = self.classifier.generate_embeddings(self.texts_small)
        self.assertEqual(len(embeddings_small), len(self.texts_small))
        
        # Test empty input
        with self.assertRaises(ValueError):
            self.classifier.generate_embeddings([])
    
    def test_model_training_balanced(self):
        """Test model training with balanced dataset."""
        # Generate embeddings
        embeddings = self.classifier.generate_embeddings(self.texts_balanced)
        
        # Train model
        self.classifier.train(embeddings, self.labels_balanced)
        
        # Check if best model is either RandomForest or LogisticRegression
        best_model = self.classifier.pipeline.named_steps['classifier']
        self.assertTrue(isinstance(best_model, (RandomForestClassifier, LogisticRegression)))
        
        # Make predictions
        predictions = self.classifier.predict(embeddings)
        
        # Check predictions
        self.assertEqual(len(predictions), len(self.labels_balanced))
        self.assertTrue(all(isinstance(p, (np.int64, int)) for p in predictions))
        
        # Check accuracy (should be better than random for balanced dataset)
        accuracy = accuracy_score(self.labels_balanced, predictions)
        self.assertGreater(accuracy, 0.6)
        
    def test_model_training_imbalanced(self):
        """Test model training with imbalanced dataset."""
        embeddings = self.classifier.generate_embeddings(self.texts_imbalanced)
        self.classifier.train(embeddings, self.labels_imbalanced)
        
        # Check predictions and class distribution
        predictions = self.classifier.predict(embeddings)
        unique_preds, pred_counts = np.unique(predictions, return_counts=True)
        
        # Should predict both classes
        self.assertEqual(len(unique_preds), 2)
        
        # Check if model handles class imbalance
        metrics = self.classifier.evaluate(embeddings, self.labels_imbalanced)
        self.assertGreater(metrics['f1_score'], 0.5)
    
    def test_model_training_small_dataset(self):
        """Test model training with very small dataset."""
        embeddings = self.classifier.generate_embeddings(self.texts_small)
        self.classifier.train(embeddings, self.labels_small)
        
        # Check if cross-validation was adjusted
        self.assertLessEqual(self.classifier.cv_results['n_splits'], min(np.bincount(self.labels_small)))
        
        # Check predictions
        predictions = self.classifier.predict(embeddings)
        self.assertEqual(len(predictions), len(self.labels_small))
    
    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        # Train on balanced dataset
        embeddings = self.classifier.generate_embeddings(self.texts_balanced)
        self.classifier.train(embeddings, self.labels_balanced)
        
        # Evaluate
        metrics = self.classifier.evaluate(embeddings, self.labels_balanced)
        
        # Check metrics structure
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('classification_report', metrics)
        self.assertIn('predictions', metrics)
        
        # Check prediction probabilities
        proba = self.classifier.predict_proba(embeddings)
        self.assertEqual(proba.shape, (len(self.texts_balanced), 2))
        self.assertTrue(np.allclose(np.sum(proba, axis=1), 1.0))
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Train model
        embeddings = self.classifier.generate_embeddings(self.texts_balanced)
        self.classifier.train(embeddings, self.labels_balanced)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.classifier.save_model(tmp_dir)
            
            # Check if files exist
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'classifier.joblib')))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'label_encoder.joblib')))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'cv_results.json')))
            
            # Create new classifier and load model
            new_classifier = PaperClassifier()
            new_classifier.load_model(tmp_dir)
            
            # Compare predictions
            orig_pred = self.classifier.predict(embeddings)
            new_pred = new_classifier.predict(embeddings)
            np.testing.assert_array_equal(orig_pred, new_pred)
            
            # Compare probabilities
            orig_proba = self.classifier.predict_proba(embeddings)
            new_proba = new_classifier.predict_proba(embeddings)
            np.testing.assert_array_almost_equal(orig_proba, new_proba)
    
    def test_cross_validation(self):
        """Test cross-validation performance."""
        # Test with balanced dataset
        embeddings = self.classifier.generate_embeddings(self.texts_balanced)
        self.classifier.train(embeddings, self.labels_balanced)
        
        # Get cross-validation scores
        cv_scores = self.classifier.get_cv_scores(embeddings, self.labels_balanced)
        
        # Check scores structure
        self.assertIsInstance(cv_scores, dict)
        self.assertIn('mean_cv_score', cv_scores)
        self.assertIn('std_cv_score', cv_scores)
        self.assertIn('individual_scores', cv_scores)
        self.assertIn('n_splits', cv_scores)
        
        # Check performance
        self.assertGreater(cv_scores['mean_cv_score'], 0.5)
        
    def test_string_labels(self):
        """Test handling of string labels."""
        # Create string labels
        string_labels = np.array(['Publishable'] * 5 + ['Non-publishable'] * 5)
        embeddings = self.classifier.generate_embeddings(self.texts_balanced)
        
        # Train with string labels
        self.classifier.train(embeddings, string_labels)
        
        # Make predictions
        predictions = self.classifier.predict(embeddings)
        self.assertTrue(all(isinstance(p, (np.int64, int)) for p in predictions))
        
        # Test prediction with new data
        new_embeddings = self.classifier.generate_embeddings(self.texts_balanced[:2])  # Get first two samples
        new_predictions = self.classifier.predict(new_embeddings)
        self.assertTrue(all(isinstance(p, (np.int64, int)) for p in new_predictions))
        
        # Evaluate with string labels
        metrics = self.classifier.evaluate(embeddings, string_labels)
        self.assertGreater(metrics['accuracy'], 0.5)
        
        # Test label encoder consistency
        encoded_labels = self.classifier.label_encoder.transform(string_labels)
        self.assertEqual(len(np.unique(encoded_labels)), 2)
        self.assertTrue(all(label in [0, 1] for label in encoded_labels))
        
        # Test inverse transform
        decoded_labels = self.classifier.label_encoder.inverse_transform(predictions)
        self.assertTrue(all(label in ['Publishable', 'Non-publishable'] for label in decoded_labels))
        
    def test_empty_input(self):
        """Test handling of empty input."""
        with self.assertRaises(ValueError) as context:
            self.classifier.generate_embeddings([])
        self.assertTrue("Cannot generate embeddings for empty text list" in str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.classifier.generate_embeddings(None)
        self.assertTrue("Cannot generate embeddings for empty text list" in str(context.exception))

if __name__ == '__main__':
    unittest.main() 
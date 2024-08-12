import unittest
import sys
import os

# Ajout du répertoire racine du projet au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.evaluate_model import evaluate_model

class TestEvaluateModel(unittest.TestCase):
    
    def setUp(self):
        # Chemins vers les fichiers de test
        self.model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models/svm_model.pkl'))
        self.test_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/sample_data_transformed_test.csv'))

    def test_evaluate_model(self):
        accuracy, report = evaluate_model(self.model_filepath, self.test_filepath)
        
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(report, str)
        
        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", report)

if __name__ == '__main__':
    unittest.main()


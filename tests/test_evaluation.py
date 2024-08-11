
import unittest
import os
import pandas as pd
from src.models.evaluate_model import evaluate_model

class TestEvaluateModel(unittest.TestCase):
    
    def setUp(self):
        # Chemin vers le fichier de modèle de test
        self.model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models/svm_model.pkl'))
        # Chemin vers le fichier de test de données
        self.test_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/mabs_transformed.csv'))

        # Vérifier que les fichiers existent
        self.assertTrue(os.path.exists(self.model_filepath), f"Model file not found: {self.model_filepath}")
        self.assertTrue(os.path.exists(self.test_filepath), f"Test data file not found: {self.test_filepath}")

    def test_evaluate_model(self):
        # Appeler la fonction d'évaluation
        accuracy, report = evaluate_model(self.model_filepath, self.test_filepath)
        
        # Vérifier que les résultats d'évaluation sont retournés
        self.assertIsInstance(accuracy, float)
        self.assertGreater(accuracy, 0, "Accuracy should be greater than 0")
        self.assertLessEqual(accuracy, 1, "Accuracy should be less or equal to 1")

        self.assertIsInstance(report, str)
        self.assertIn("Fake_news", report)
        self.assertIn("Good_news", report)
        
        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", report)

if __name__ == '__main__':
    unittest.main()


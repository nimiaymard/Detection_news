import os
import pandas as pd
import unittest

# Classe pour les tests unitaires
class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Les fichiers de test existent déjà
        self.input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw_fake_news/sample_data_test.csv'))
        self.output_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed_fake_news/sample_data_transformed_test.csv'))

    def test_load_and_transform_data(self):
        # Vérifier que les fichiers de test existent déjà
        self.assertTrue(os.path.exists(self.input_filepath))
        self.assertTrue(os.path.exists(self.output_filepath))
        
        # Charger les données transformées
        data_transformed = pd.read_csv(self.output_filepath)
        
        # Vérifier que les colonnes attendues existent
        self.assertIn('annotations_num1', data_transformed.columns)

        # Vérifier que les données sont correctement transformées
        self.assertEqual(data_transformed['annotations_num1'].iloc[0], 0)
        self.assertEqual(data_transformed['annotations_num1'].iloc[1], 1)
        self.assertTrue(pd.isna(data_transformed['annotations_num1'].iloc[2]))

if __name__ == '__main__':
    # Exécution des tests unitaires
    unittest.main(exit=False)

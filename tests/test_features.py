import unittest
import os
import pandas as pd

class TestFeatures(unittest.TestCase):

    def setUp(self):
        # Chemins vers les fichiers de test déjà existants
        self.input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/sample_data_test.csv'))
        self.output_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/sample_data_transformed_test.csv'))

    def test_load_and_transform_data(self):
        # Chargement des données transformées
        data_transformed = pd.read_csv(self.output_filepath)

        # Vérifier que la colonne 'annotations_num1' existe
        self.assertIn('annotations_num1', data_transformed.columns)

        # Vérification des données dans 'annotations_num1' sont correctement mappées
        self.assertEqual(data_transformed['annotations_num1'].iloc[0], 0)  # 'Fake_news' est mappé à 0
        self.assertEqual(data_transformed['annotations_num1'].iloc[1], 1)  # 'Good_news' est mappé à 1
        self.assertTrue(pd.isna(data_transformed['annotations_num1'].iloc[2]))  # None reste NaN

        # Vérification des fonctionnalités spécifiques si elles existent
        if 'new_feature' in data_transformed.columns:
            self.assertGreater(data_transformed['new_feature'].sum(), 0)

    def tearDown(self):
        # Suppression des fichiers de test après les tests si nécessaire
        if os.path.exists(self.input_filepath):
            os.remove(self.input_filepath)
        if os.path.exists(self.output_filepath):
            os.remove(self.output_filepath)

if __name__ == '__main__':
    unittest.main()



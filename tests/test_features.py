import unittest
import os
import pandas as pd
from src.features.build_features import load_and_transform_data

class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/mabs_test.csv'))
        self.output_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/mabs_transformed_test.csv'))

        data = {
            'Title_news': ['News 1', 'News 2', 'News 3'],
            'Annotations': ['Fake_news', 'Good_news', None]
        }
        self.test_data = pd.DataFrame(data)
        self.test_data.to_csv(self.input_filepath, index=False)

    def test_load_and_transform_data(self):
        load_and_transform_data(self.input_filepath, self.output_filepath)
        data_transformed = pd.read_csv(self.output_filepath)

        # Vérifier des fonctionnalités spécifiques si elles existent
        self.assertIn('new_feature', data_transformed.columns)
        self.assertGreater(data_transformed['new_feature'].sum(), 0)

    def tearDown(self):
        if os.path.exists(self.input_filepath):
            os.remove(self.input_filepath)
        if os.path.exists(self.output_filepath):
            os.remove(self.output_filepath)

if __name__ == '__main__':
    unittest.main()


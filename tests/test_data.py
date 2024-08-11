import pandas as pd
import os
import unittest

# Préparer et enregistrer le DataFrame de test
def prepare_test_data():
    # Création d'un petit DataFrame pour les tests
    data = {
        'Title_news': ['News 1', 'News 2', 'News 3'],
        'Annotations': ['Fake_news', 'Good_news', None]
    }
    df = pd.DataFrame(data)

    # Chemin pour enregistrer les données de test
    test_data_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/mabs_test.csv'))
    
    # Créer le répertoire s'il n'existe pas
    test_data_dir = os.path.dirname(test_data_filepath)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    
    # Sauvegarder le DataFrame de test en tant que fichier CSV
    df.to_csv(test_data_filepath, index=False)
    print(f"Test data saved to {test_data_filepath}")

    return test_data_filepath

# Fonction pour transformer les données
def load_and_transform_data(input_filepath, output_filepath):
    # Chargement des données
    data = pd.read_csv(input_filepath)
    
    # Mapping des annotations textuelles en numériques
    annotation_map = {'Fake_news': 0, 'Good_news': 1}
    data['annotations_num1'] = data['Annotations'].map(annotation_map)
    
    # Création du répertoire de sortie s'il n'existe pas
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sauvegarde du DataFrame transformé en fichier CSV
    data.to_csv(output_filepath, index=False)
    print(f"DataFrame transformé sauvegardé dans {output_filepath}")

# Classe pour les tests unitaires
class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Chemins vers les fichiers de test
        self.input_filepath = prepare_test_data()
        self.output_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/mabs_transformed_test.csv'))

    def test_load_and_transform_data(self):
        # Appeler la fonction de transformation des données
        load_and_transform_data(self.input_filepath, self.output_filepath)
        
        # Vérifier que le fichier de sortie est créé
        self.assertTrue(os.path.exists(self.output_filepath))
        
        # Charger les données transformées
        data_transformed = pd.read_csv(self.output_filepath)
        
        # Vérifier que les colonnes attendues existent
        self.assertIn('annotations_num1', data_transformed.columns)

        # Vérifier que les données sont correctement transformées
        self.assertEqual(data_transformed['annotations_num1'].iloc[0], 0)
        self.assertEqual(data_transformed['annotations_num1'].iloc[1], 1)
        self.assertTrue(pd.isna(data_transformed['annotations_num1'].iloc[2]))

    def tearDown(self):
        # Supprimer les fichiers de test après les tests
        if os.path.exists(self.output_filepath):
            os.remove(self.output_filepath)

if __name__ == '__main__':
    # Exécution des tests unitaires
    unittest.main(exit=False)
    
    # Chemin d'entrée (fichier de données d'origine)
    input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/mabs.csv'))
    
    # Chemin de sortie (fichier de données transformées)
    output_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/mabs_transformed.csv'))
    
    # Appel de la fonction de transformation
    load_and_transform_data(input_filepath, output_filepath)


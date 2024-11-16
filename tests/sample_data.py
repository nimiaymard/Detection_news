import pandas as pd
import os

def prepare_and_transform_data():
    # Étape 1: Création d'un petit DataFrame pour les tests
    data = {
        'Title_news': ['News 1', 'News 2', 'News 3'],
        'Annotations': ['Fake_news', 'Good_news', None]
    }
    df = pd.DataFrame(data)

    # Chemin pour enregistrer les données de test dans data/raw
    raw_data_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw_fake_news/sample_data_test.csv'))
    
    # Créer le répertoire s'il n'existe pas
    raw_data_dir = os.path.dirname(raw_data_filepath)
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
    
    # Sauvegarder le DataFrame de test en tant que fichier CSV
    df.to_csv(raw_data_filepath, index=False)
    print(f"Test data saved to {raw_data_filepath}")

    # Étape 2: Transformation des données
    # Mapping des annotations textuelles en valeurs numériques
    annotation_map = {'Fake_news': 0, 'Good_news': 1}
    df['annotations_num1'] = df['Annotations'].map(annotation_map)

    # Chemin pour enregistrer les données transformées dans data/processed
    transformed_data_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed_fake_news/sample_data_transformed_test.csv'))
    
    # Créer le répertoire s'il n'existe pas
    transformed_data_dir = os.path.dirname(transformed_data_filepath)
    if not os.path.exists(transformed_data_dir):
        os.makedirs(transformed_data_dir)
    
    # Sauvegarder le DataFrame transformé en tant que fichier CSV
    df.to_csv(transformed_data_filepath, index=False)
    print(f"Transformed data saved to {transformed_data_filepath}")

    return raw_data_filepath, transformed_data_filepath

if __name__ == '__main__':
    # Préparer, transformer et sauvegarder les données
    prepare_and_transform_data()

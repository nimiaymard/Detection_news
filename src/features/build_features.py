import pandas as pd
import os

def load_and_transform_data(input_filepath, output_filepath):
    # Chargement des données
    data = pd.read_csv(input_filepath)
    
    # Data_without_missing est déjà défini
    data_without_missing = data[data['Annotations'].notna()].copy()

    # Conversion des annotations textuelles en numériques
    annotation_map = {'Fake_news': 0, 'Good_news': 1}
    data_without_missing['annotations_num1'] = data_without_missing['Annotations'].map(annotation_map)
    
    # Séparation des données en fonctionnalités (X) et cibles (y)
    X = data_without_missing['Title_news']
    y = data_without_missing['Annotations']
    
    # Enregistrement du DataFrame en fichier CSV dans le répertoire processed
    data_without_missing.to_csv(output_filepath, index=False)

if __name__ == "__main__":
    # Chemin d'entrée
    input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed_fake_news/mabs.csv'))
    
    # Chemin de sortie
    output_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed_fake_news/mabs_transformed.csv'))
    
    load_and_transform_data(input_filepath, output_filepath)

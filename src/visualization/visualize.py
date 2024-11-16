import os
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from plot_results import plot_class_distribution, plot_confusion_matrix, plot_classification_report
import pandas as pd

def load_data(input_filepath):
    # Chargement des données transformées
    data = pd.read_csv(input_filepath)
    return data

def evaluate_and_visualize(input_filepath, model_filepath):
    # Chargement des données
    data = load_data(input_filepath)
    
    # Visualisation de la distribution des classes
    plot_class_distribution(data)

    # Chargement du modèle et du vectoriseur
    svm, vectorizer = joblib.load(model_filepath)

    # Séparation des fonctionnalités (X) et la cible (y)
    X = data['Title_news']
    y = data['annotations_num1']

    # Vectorisation des données textuelles
    X_vectorized = vectorizer.transform(X)

    # Prédire les étiquettes avec le modèle chargé
    y_pred = svm.predict(X_vectorized)

    # Visualisation de la matrice de confusion
    plot_confusion_matrix(y, y_pred)

    # Visualisation du rapport de classification
    plot_classification_report(y, y_pred)

if __name__ == "__main__":
    # Chemin d'entrée
    input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed_fake_news/mabs_transformed.csv'))
    # Chemin du modèle sauvegardé
    model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_models/svm_model.pkl'))
    
    # Évaluation et visualisation des résultats
    evaluate_and_visualize(input_filepath, model_filepath)

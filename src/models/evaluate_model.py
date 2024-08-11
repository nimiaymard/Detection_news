import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from train_model import load_data, preprocess_and_vectorize
def evaluate_model(model_filepath, input_filepath):
    # Charger les données

    data = load_data(input_filepath)
    
    # Prétraiter et vectoriser les données
    X_vectorized, y, vectorizer = preprocess_and_vectorize(data)
    
    # Chargement du modèle et du vectoriseur
    model, _ = joblib.load(model_filepath)
    
    # Division des données en ensembles d'entraînement et de test
    test_size = 0.3  # 30% des données seront utilisées pour le test
    _, X_test, _, y_test = train_test_split(X_vectorized, y, test_size=test_size, random_state=42)
    
    # Prédictions sur les données de test
    y_pred = model.predict(X_test)
    
    # Calcul de l'accuracy et génération d'un rapport de classification
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Fake_news', 'Good_news'])
    
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    # Chemin du fichier du modèle
    model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_models/svm_model.pkl'))
    # Chemin du fichier des données
    input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/mabs_transformed.csv'))
    
    # Évaluation de modèle
    evaluate_model(model_filepath, input_filepath)



import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from .train_model import load_data, preprocess_and_vectorize

def evaluate_model(model_filepath, input_filepath):
    # Chargement des données
    data = load_data(input_filepath)
    
    # Prétraitement et vectorisation des données
    X_vectorized, y, vectorizer = preprocess_and_vectorize(data)
    
    # Chargement du modèle et du vectoriseur
    model, _ = joblib.load(model_filepath)
    
    # Validation croisée
    scores = cross_val_score(model, X_vectorized, y, cv=5)
    print(f"Validation croisée (k=5) scores: {scores}")
    print(f"Accuracy moyenne: {scores.mean():.3f}")
    print(f"Écart-type des accuracy: {scores.std():.3f}")
    
    # Prédictions sur l'ensemble complet (sans réentraînement)
    y_pred = model.predict(X_vectorized)
    
    # Calcul de l'accuracy et génération d'un rapport de classification
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=['Fake_news', 'Good_news'])
    
    print("\n--- Rapport de classification complet ---")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    # Chemin du fichier du modèle
    model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_models/svm_model.pkl'))
    # Chemin du fichier des données
    input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/mabs_transformed.csv'))
    
    # Évaluation de modèle
    evaluate_model(model_filepath, input_filepath)


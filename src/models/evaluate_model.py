
#from .train_model import load_data, preprocess_and_vectorize

import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

def evaluate_model(model_filepath, input_filepath):
    try:
        # Chargement des données
        data = pd.read_csv(input_filepath)
        print("Données chargées avec succès.")
    except FileNotFoundError:
        print(f"Erreur : fichier introuvable à {input_filepath}")
        return

    try:
        # Chargement du modèle et du vectoriseur
        model, vectorizer = joblib.load(model_filepath)
        print("Modèle et vectoriseur chargés avec succès.")
    except FileNotFoundError:
        print(f"Erreur : fichier modèle introuvable à {model_filepath}")
        return

    # Prétraitement et vectorisation des données
    try:
        X_vectorized = vectorizer.transform(data['Title_news'])
        y = data['annotations_num1']
        print("Prétraitement et vectorisation des données terminé.")
    except KeyError as e:
        print(f"Erreur de prétraitement : colonne manquante {e}")
        return

    # Validation croisée
    print("Évaluation avec validation croisée...")
    scores = cross_val_score(model, X_vectorized, y, cv=5, n_jobs=-1)
    print(f"Validation croisée (k=5) scores: {scores}")
    print(f"Accuracy moyenne: {scores.mean():.3f}")
    print(f"Écart-type des accuracy: {scores.std():.3f}")
    
    # Prédictions sur l'ensemble complet
    print("Prédictions sur l'ensemble complet...")
    y_pred = model.predict(X_vectorized)
    
    # Calcul des métriques
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=['Fake_news', 'Good_news'])
    
    print("\n--- Rapport de classification complet ---")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)


if __name__ == "__main__":
    # Chemin du fichier modèle
    model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_models/svm_model.pkl'))
    
    # Chemin des données d'évaluation
    input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed_fake_news/mabs_transformed.csv'))
    
    # Évaluation de modèle
    evaluate_model(model_filepath, input_filepath)

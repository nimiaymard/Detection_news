import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import os
import joblib

def load_data(input_filepath):
    try:
        # Chargement des données
        data = pd.read_csv(input_filepath)
        print("Données chargées avec succès.")
        return data
    except FileNotFoundError:
        print(f"Erreur : fichier introuvable à {input_filepath}")
        exit(1)

def preprocess_and_vectorize(data):
    # Vérification des colonnes nécessaires
    if data['Title_news'].isnull().any() or data['annotations_num1'].isnull().any():
        print("Attention : des valeurs manquantes ont été détectées et supprimées.")
        data = data.dropna(subset=['Title_news', 'annotations_num1'])

    # Séparation des fonctionnalités (X) et de la cible (y)
    X = data['Title_news']
    y = data['annotations_num1']
    
    # Vectorisation des données textuelles
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    
    return X_vectorized, y, vectorizer

def perform_grid_search(X_train, y_train):
    # Définir une grille d'hyperparamètres
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    # Créer un modèle SVM avec gestion des classes déséquilibrées
    svm = SVC(class_weight='balanced')

    # Recherche d'hyperparamètres
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_

def train_svm(X, y):
    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Recherche d'hyperparamètres
    print("Recherche d'hyperparamètres...")
    best_svm, best_params = perform_grid_search(X_train, y_train)

    print(f"Meilleurs hyperparamètres trouvés : {best_params}")
    

    return best_svm, vectorizer, X_test, y_test

def save_model(model, vectorizer, model_filepath):
    # Sauvegarde du modèle
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    joblib.dump((model, vectorizer), model_filepath)
    print(f"Modèle et vectoriseur sauvegardés à {model_filepath}")

if __name__ == "__main__":
    input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed_fake_news/mabs_transformed.csv'))
    data = load_data(input_filepath)
    X_vectorized, y, vectorizer = preprocess_and_vectorize(data)
    svm, vectorizer, X_test, y_test = train_svm(X_vectorized, y)
    model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_models/svm_model.pkl'))
    save_model(svm, vectorizer, model_filepath)

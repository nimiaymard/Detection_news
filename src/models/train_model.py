import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib  # Pour sauvegarder et charger le modèle

def load_data(input_filepath):
    # Chargement des données transformées
    data = pd.read_csv(input_filepath)
    return data

def preprocess_and_vectorize(data):
    # Séparation des fonctionnalités (X) et la cible (y)
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
    
    # Créer un modèle SVM
    svm = SVC()

    # Effectuer la recherche d'hyperparamètres avec GridSearchCV
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Retourner le meilleur modèle et les meilleurs hyperparamètres
    return grid_search.best_estimator_, grid_search.best_params_

def train_svm(X, y):
    # Division des données en ensembles d'entraînement et de test
    test_size = 0.3  # 30% des données seront utilisées pour le test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Effectuer la recherche d'hyperparamètres et obtenir le meilleur modèle
    best_svm, best_params = perform_grid_search(X_train, y_train)

    print(f"Meilleurs hyperparamètres trouvés : {best_params}")
    
    # Entraîner le modèle avec les meilleurs hyperparamètres
    best_svm.fit(X_train, y_train)
    
    # Retourner le modèle entraîné, le vectoriseur et les ensembles de test pour une future évaluation
    return best_svm, vectorizer, X_test, y_test

def save_model(model, vectorizer, model_filepath):
    # Sauvegarde du modèle entraîné et du vectoriseur
    joblib.dump((model, vectorizer), model_filepath)
    print(f"Model and vectorizer saved to {model_filepath}")

if __name__ == "__main__":
    # Chemin d'entrée
    input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/mabs_transformed.csv'))

    # Chargement des données
    data = load_data(input_filepath)
    
    # Prétraitement et vectorisation des données
    X_vectorized, y, vectorizer = preprocess_and_vectorize(data)
    
    # Entraînement du modèle SVM avec les meilleurs hyperparamètres
    svm, vectorizer, X_test, y_test = train_svm(X_vectorized, y)

    # Chemin pour sauvegarder le modèle
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../saved_models'))
    model_filepath = os.path.join(model_dir, 'svm_model.pkl')
    
    # Création du répertoire s'il n'existe pas
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Sauvegarde du modèle et du vectoriseur
    save_model(svm, vectorizer, model_filepath)

import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import learning_curve, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Chargement du modèle et le vectoriseur
model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models/svm_model.pkl'))
model, vectorizer = joblib.load(model_filepath) 

# Définir la configuration de la page pour un affichage en pleine largeur
st.set_page_config(layout="wide")

# Centrer le titre de l'application
st.markdown("<h1 style='text-align: center;'>Détection de Fake News</h1>", unsafe_allow_html=True)

# Fonction pour tracer la courbe d'apprentissage
def plot_learning_curve(estimator, X, y, ax=None):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, n_jobs=-1, 
                                                            train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    if ax is None:
        ax = plt.gca()
    
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation croisée")
    ax.set_title('Courbe d\'apprentissage')
    ax.set_xlabel('Entraînement')
    ax.set_ylabel('Score')
    ax.legend(loc='best')
    return ax

# Fonction pour prédire les étiquettes à partir de nouvelles données
def predict(input_data):
    # Utilisation du vectoriseur chargé depuis le fichier pkl
    X_vectorized = vectorizer.transform(input_data)
    predictions = model.predict(X_vectorized)
    
    label_map = {0: 'Fake_news', 1: 'Good_news'}
    predictions = [label_map[pred] for pred in predictions]
    
    return predictions

# Section pour télécharger les données
uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Aperçu des données:")
    st.dataframe(data.head())

    if st.button("Prédire"):
        if 'Title_news' in data.columns:
            predictions = predict(data['Title_news'])
            data['Predictions'] = predictions
            
            left_column, right_column = st.columns(2)
            with left_column:
                st.write("Résultats de la prédiction:")
                st.dataframe(data[['Title_news', 'Predictions']])
            
            st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
            
            with right_column:
                st.write("Rapport de classification:")
                if 'Annotations' in data.columns:
                    report = classification_report(data['Annotations'], data['Predictions'], target_names=['Fake_news', 'Good_news'])
                    st.text(report)
                    
                    X = vectorizer.transform(data['Title_news'])
                    accuracy_cv = cross_val_score(model, X, data['Annotations'], cv=5, scoring='accuracy')
                    st.write(f"Accuracy (Validation Croisée 5-plis): {accuracy_cv.mean():.3f} ± {accuracy_cv.std():.3f}")
                else:
                    st.error("La colonne 'Annotations' est manquante dans le fichier CSV.")
        
        left_column, right_column = st.columns(2)
        with left_column:
            st.write("Matrice de confusion:")
            if 'Annotations' in data.columns:
                cm = confusion_matrix(data['Annotations'], data['Predictions'], labels=['Fake_news', 'Good_news'])
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_title('Matrice de Confusion')
                st.pyplot(fig_cm)
            else:
                st.error("La colonne 'Annotations' est manquante dans le fichier CSV.")
                
        # Ajouter un espace vertical entre la matrice de confusion et la courbe d'apprentissage
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        
        with right_column:
            st.write("Courbe d'apprentissage:")
            if 'Annotations' in data.columns:
                fig_lc, ax_lc = plt.subplots()
                X = vectorizer.transform(data['Title_news'])
                plot_learning_curve(model, X, data['Annotations'], ax_lc)
                st.pyplot(fig_lc)
            else:
                st.error("La colonne 'Annotations' est manquante dans le fichier CSV.")
    else:
        st.error("Vous devez d'abord effectuer une prédiction.")


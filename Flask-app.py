from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Chargement du modèle et du vectoriseur
model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models/svm_model.pkl'))
model, vectorizer = joblib.load(model_filepath)

# Fonction pour prédire les étiquettes
def predict(input_data):
    X_vectorized = vectorizer.transform(input_data)
    predictions = model.predict(X_vectorized)
    label_map = {0: 'Fake_news', 1: 'Good_news'}
    predictions = [label_map[pred] for pred in predictions]
    return predictions

# Fonction pour tracer la courbe d'apprentissage
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation croisée")
    plt.title('Courbe d\'apprentissage')
    plt.xlabel('Taille de l\'ensemble d\'entraînement')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('static/learning_curve.png')
    plt.close()

# Page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour gérer l'upload de fichier
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('home', error="Aucun fichier sélectionné"))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home', error="Le fichier est vide"))

    # Lire le fichier CSV
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return redirect(url_for('home', error=f"Erreur lors du traitement du fichier : {e}"))

    if 'Title_news' not in data.columns:
        return redirect(url_for('home', error="La colonne 'Title_news' est manquante."))

    # Prédiction
    data['Predictions'] = predict(data['Title_news'])

    # Rapport de classification (si les annotations sont présentes)
    classification_report_text = ""
    if 'Annotations' in data.columns:
        report = classification_report(data['Annotations'], data['Predictions'], target_names=['Fake_news', 'Good_news'])
        classification_report_text = report

        # Matrice de confusion
        cm = confusion_matrix(data['Annotations'], data['Predictions'], labels=['Fake_news', 'Good_news'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake_news', 'Good_news'], yticklabels=['Fake_news', 'Good_news'])
        plt.title('Matrice de Confusion')
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité')
        plt.tight_layout()
        plt.savefig('static/confusion_matrix.png')
        plt.close()

        # Courbe d'apprentissage
        X_vectorized = vectorizer.transform(data['Title_news'])
        plot_learning_curve(model, X_vectorized, data['Annotations'])

    # Sauvegarde temporaire du CSV avec les prédictions
    output_path = 'static/results.csv'
    data.to_csv(output_path, index=False)

    return render_template(
        'results.html',
        preview=data.head().to_html(),
        classification_report=classification_report_text,
        results_csv=output_path
    )

if __name__ == '__main__':
    app.run(debug=True)

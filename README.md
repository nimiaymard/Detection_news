# Projet Académique : Détection de Fake News

Ce projet académique explore l'utilisation de techniques de machine learning pour la détection automatique des fake news. Le projet comprend l'exploration des données, la création de features, l'entraînement de modèles, et la visualisation des résultats. 

## Objectif du Projet

L'objectif principal de ce projet est de développer un modèle de machine learning capable de classifier des articles de presse en tant que "Fake News" ou "Good News". Ce projet vise à appliquer des méthodes d'apprentissage supervisé, en particulier les Support Vector Machines (SVM), pour aborder ce problème.

## Structure du Projet

- `data/` : Contient les données brutes et les données prétraitées pour l'analyse.
- `notebooks/` : Contient les notebooks Jupyter pour l'exploration des données, le prétraitement, et le prototypage.
- `src/` : Contient les scripts Python organisés par fonction (ex. : prétraitement, modélisation, visualisation).
- `tests/` : Contient les tests unitaires pour vérifier le bon fonctionnement des différents composants du projet.
- `saved_models/` : Contient les modèles entraînés qui ont été sauvegardés pour une utilisation future.
- `requirements.txt` : Liste des bibliothèques Python nécessaires pour exécuter le projet.
- `Dockerfile` et `docker-compose.yml` : Fichiers pour la configuration d'un environnement Docker (optionnel).
- `Makefile` : Fichier pour automatiser les tâches courantes (optionnel).
- `.gitignore` : Liste des fichiers et répertoires à ignorer par Git.
- `README.md` : Ce fichier, qui décrit le projet.
- `setup.py` : Script pour installer le projet comme un module Python.

## Contexte

Dans un contexte où la propagation de la désinformation est un problème croissant, ce projet propose une solution basée sur le machine learning pour identifier les fausses informations dans les médias. Le projet se concentre sur l'application de méthodes d'apprentissage supervisé et sur l'évaluation de leur performance.

## Installation

1. **Clonez le dépôt :**
    ```bash
    git clone https://github.com/nimiaymard/Detection_news.git
    cd my_ml_project
    ```

2. **Créez et activez un environnement virtuel :**
   - Sous Windows :
     ```bash
     python -m venv myenv
     myenv\Scripts\activate
     ```
   - Sous macOS/Linux :
     ```bash
     python3 -m venv myenv
     source myenv/bin/activate
     ```

3. **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### Prétraitement des Données
Pour charger et transformer les données brutes:
```bash
python src/features/build_features.py
```

### Entrainement du modèle
Pour entraîner le modèle SVM sur les données préparées:
```bash
python src/models/train_model.py
```
### Evaluation du modèle
Pour évaluer les performances du modèle 
```bash
python src/models/evaluate_model.py
```
### Visualisation des résultats
Pour visualiser la distribution des classes et les métriques de performance:
```bash
python src/visualization/plot_results.py
```
### Tests
Pour exécuter les tests unitaires:
```bash
pytest tests/
```

### Méthodologie
Ce projet suit les étapes classiques de l'apprentissage supervisé :
1. Collecte des données : Importation des données à partir de diverses sources.
Prétraitement des données : Nettoyage des données, gestion des valeurs manquantes, et vectorisation des données textuelles.
2. Modélisation : Entraînement d'un modèle SVM pour la classification des nouvelles.
3. Évaluation : Utilisation de métriques telles que la précision, le rappel et le score F1 pour évaluer la performance du modèle.
4. Visualisation : Utilisation de graphiques pour illustrer la performance du modèle.

### Conclusion
Ce projet montre comment les techniques de machine learning peuvent être utilisées pour résoudre des problèmes complexes comme la détection des fake news. Les résultats obtenus montrent que le modèle SVM est capable de classifier les nouvelles avec un bon niveau de précision.

### Licence
Ce projet est réalisé dans un cadre académique et est mis à disposition sous la licence MIT. Pour plus de détails, voir le fichier LICENSE.

### Remerciements
Merci à Ben pour les conseils et le soutien tout au long de ce projet.




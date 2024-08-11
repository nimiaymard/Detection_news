import unittest
import os
import matplotlib.pyplot as plt
from src.visualization.visualize import plot_confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix

class TestVisualization(unittest.TestCase):

    def setUp(self):
        # Créer un répertoire temporaire pour sauvegarder les graphiques
        self.test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reports/figures'))
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        # Données fictives pour le test
        self.y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        self.y_pred = [0, 0, 0, 1, 0, 1, 1, 1]
        self.cm = confusion_matrix(self.y_true, self.y_pred)
        self.labels = ['Fake_news', 'Good_news']

    def test_plot_confusion_matrix(self):
        # Chemin du fichier de sortie
        output_filepath = os.path.join(self.test_dir, 'confusion_matrix.png')

        # Appeler la fonction de traçage de la matrice de confusion
        plot_confusion_matrix(self.cm, classes=self.labels, output_filepath=output_filepath)

        # Vérifier que le fichier de sortie est créé
        self.assertTrue(os.path.exists(output_filepath))

    def tearDown(self):
        # Supprimer les fichiers de test après les tests
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                file_path = os.path.join(self.test_dir, file)
                os.remove(file_path)
            os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main()

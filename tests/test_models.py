import unittest
import os
from src.models.train_model import preprocess_and_vectorize, train_svm, save_model, load_model

class TestModels(unittest.TestCase):
    
    def setUp(self):
        # Chemin vers le fichier d'entrée de test
        self.input_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/sample_data_transformed_test.csv'))
        # Chemin vers le fichier de modèle de test
        self.model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../saved_models/svm_model_test.pkl'))

    def test_preprocess_and_vectorize(self):
        # Appeler la fonction de prétraitement et vectorisation
        X_vectorized, y, vectorizer = preprocess_and_vectorize(self.input_filepath)
        
        # Vérifier que la vectorisation a été effectuée
        self.assertGreater(X_vectorized.shape[0], 0)
        self.assertGreater(X_vectorized.shape[1], 0)
        self.assertEqual(len(y), X_vectorized.shape[0])

    def test_train_svm(self):
        # Prétraiter et vectoriser les données
        X_vectorized, y, vectorizer = preprocess_and_vectorize(self.input_filepath)
        
        # Entraîner le modèle
        svm, vectorizer, X_test, y_test = train_svm(X_vectorized, y)
        
        # Vérifier que le modèle a été entraîné
        self.assertIsNotNone(svm)

    def test_save_and_load_model(self):
        # Prétraiter et vectoriser les données
        X_vectorized, y, vectorizer = preprocess_and_vectorize(self.input_filepath)
        
        # Entraîner le modèle
        svm, vectorizer, X_test, y_test = train_svm(X_vectorized, y)
        
        # Sauvegarder le modèle
        save_model(svm, vectorizer, self.model_filepath)
        
        # Vérifier que le fichier de modèle est créé
        self.assertTrue(os.path.exists(self.model_filepath))
        
        # Charger le modèle
        loaded_svm, loaded_vectorizer = load_model(self.model_filepath)
        
        # Vérifier que le modèle et le vectoriseur sont chargés correctement
        self.assertIsNotNone(loaded_svm)
        self.assertIsNotNone(loaded_vectorizer)

    def tearDown(self):
        # Supprimer le fichier de modèle après le test
        if os.path.exists(self.model_filepath):
            os.remove(self.model_filepath)

if __name__ == '__main__':
    unittest.main()

import os
import shutil
import numpy as np

# Chemin vers le dossier contenant les images
base_dir = './dataset_images'  # Chemin de base
train_dir = os.path.join(base_dir, 'train')  # Dossier d'entraînement
validation_dir = os.path.join(base_dir, 'validation')  # Dossier de validation

# Création des sous-dossiers pour 'cats' et 'dogs' dans 'train' et 'validation'
for folder in ['cats', 'dogs']:
    os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, folder), exist_ok=True)

# Gestion des sous-dossiers source ('cat' et 'dog')
categories = {'cat': 'cats', 'dog': 'dogs'}
all_images = []  # Liste de toutes les images

# Parcourir les dossiers source et collecter les chemins des fichiers
for src_folder, dest_folder in categories.items():
    category_path = os.path.join(base_dir, src_folder)
    if not os.path.exists(category_path):
        raise FileNotFoundError(f"Le dossier source n'existe pas : {category_path}")

    images = [os.path.join(category_path, img) for img in os.listdir(category_path)
              if img.endswith(('.jpg', '.png', '.jpeg'))]
    all_images.extend([(img, dest_folder) for img in images])  # Associe chaque image à sa catégorie cible

# Mélanger aléatoirement les fichiers
np.random.shuffle(all_images)

# Diviser en ensembles train/validation (80% pour train, 20% pour validation)
split_index = int(0.8 * len(all_images))
train_images = all_images[:split_index]
validation_images = all_images[split_index:]

# Déplacer les fichiers dans les dossiers correspondants
for img_path, dest_folder in train_images:
    dest_path = os.path.join(train_dir, dest_folder, os.path.basename(img_path))
    shutil.move(img_path, dest_path)

for img_path, dest_folder in validation_images:
    dest_path = os.path.join(validation_dir, dest_folder, os.path.basename(img_path))
    shutil.move(img_path, dest_path)

print("Répartition terminée avec succès.")


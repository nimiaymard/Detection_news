import os

# Chemins de base
base_dir = './dataset_images'
sub_dirs = ['train', 'validation']
categories = ['cats', 'dogs']

# Création des dossiers
for sub in sub_dirs:
    for category in categories:
        # Créer le chemin du dossier
        new_dir = os.path.join(base_dir, sub, category)
        # Si le dossier n'existe pas, créez-le
        os.makedirs(new_dir, exist_ok=True)

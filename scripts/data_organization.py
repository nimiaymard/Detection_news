import shutil
import numpy as np

# Chemin vers votre dossier actuel d'images
path_to_images = './dataset_images'
images = os.listdir(path_to_images)

# Mélanger aléatoirement les images pour une répartition équitable
np.random.shuffle(images)

# Répartition: 80% pour l'entraînement, 20% pour la validation
split = int(len(images) * 0.8)

# Déplacer les images
for i, image in enumerate(images):
    if 'cat' in image:
        folder = 'cats'
    else:
        folder = 'dogs'

    if i < split:
        shutil.move(os.path.join(path_to_images, image), os.path.join(base_dir, 'train', folder, image))
    else:
        shutil.move(os.path.join(path_to_images, image), os.path.join(base_dir, 'validation', folder, image))

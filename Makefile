# Nom du conteneur Docker
CONTAINER_NAME=streamlit_fake_news

# Nom de l'image Docker
IMAGE_NAME=fake_news_app

# Commandes pour les différentes cibles

# Construire l'image Docker
build:
    docker build -t $(IMAGE_NAME) .

# Lancer les conteneurs avec Docker Compose
up:
    docker-compose up

# Arrêter les conteneurs
down:
    docker-compose down

# Exécuter les tests
test:
    python -m unittest discover -s tests

# Déploiement de l'application Streamlit
deploy:
    docker-compose up --build -d

# Nettoyer les conteneurs et images Docker inutilisés
clean:
    docker system prune -f

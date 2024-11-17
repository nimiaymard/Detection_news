// Script to add some interactivity

// Afficher un message lorsque l'utilisateur télécharge un fichier
document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        fileInput.addEventListener('change', function () {
            if (this.files.length > 0) {
                alert(`Fichier sélectionné : ${this.files[0].name}`);
            }
        });
    }
});

// Confirmation avant d'envoyer le formulaire
document.querySelector('form').addEventListener('submit', function (e) {
    const confirmation = confirm("Êtes-vous sûr de vouloir envoyer ce fichier ?");
    if (!confirmation) {
        e.preventDefault(); // Annule l'envoi si l'utilisateur annule
    }
});

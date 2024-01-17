import streamlit as st  # Import de la bibliothèque Streamlit pour créer l'interface utilisateur
from PIL import Image  # Import de la bibliothèque PIL pour manipuler les images
import tensorflow as tf  # Import de la bibliothèque TensorFlow pour charger le modèle VGG19
import numpy as np  # Import de la bibliothèque NumPy pour manipuler les tableaux
from tensorflow.keras.applications.vgg19 import preprocess_input  # Import de la fonction de prétraitement de VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Import de fonctions pour charger et convertir les images
import os  # Import de la bibliothèque os pour manipuler les fichiers
import requests

# Charger le modèle VGG19 pré-entraîné

def download_file_from_url(url, destination):
    response = requests.get(url)

    with open(destination, "wb") as file:
        file.write(response.content)

def load_model():
    model_path = "model.h5"  # Path to save the downloaded model file
    model_url = "https://www.dropbox.com/scl/fi/hyqdtu8gckz1l4483weps/VGG19_TeamMaroc.h5?rlkey=4cgeps6umtruvs79wwddej6vy&dl=1"  # URL of the model file

    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            download_file_from_url(model_url, model_path)

    model = tf.keras.models.load_model(model_path)

    return model
    
model = load_model()
#model = tf.keras.models.load_model(st.secrets["PATH"])

# Définir les étiquettes de classe
class_labels = ['Achraf Hakimi', 'Azzedine Ounahi', 'Hakim Ziyech', 'Nayef Aguerd', 'Noussair Mazraoui',
                'Romain Saiss', 'Selim Amallah', 'Soufyan Amrabat', 'Soufyan Boufal', 'Yassin Bono', 'Youssef Ennesyri']

def predict_image_class(image_path, model, class_labels):
    # Charger et prétraiter l'image à classifier
    image_size = (224, 224)  # Ajuster la taille en fonction de la taille d'entrée du modèle
    image = load_img(image_path, target_size=image_size)
    image_array = np.expand_dims(img_to_array(image) / 255, axis=0)

    # Faire les prédictions
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions[0])

    # Obtenir la classe prédite
    predicted_class = class_labels[predicted_class_index]

    # Retourner la classe prédite
    return predicted_class

def main():
    st.title('Reconnaissance faciale avec VGG19')  # Titre de l'application
    uploaded_file = st.file_uploader('Charger une image', type=['jpg', 'jpeg', 'png'])  # Zone de téléchargement du fichier image

    if uploaded_file is not None:
        # Lire l'image téléchargée
        image = Image.open(uploaded_file)
        st.image(image, caption='Charger une image', use_column_width=False)  # Afficher l'image sur l'interface

        # Sauvegarder l'image téléchargée dans un fichier temporaire
        temp_file = f"temp.{uploaded_file.name.split('.')[-1]}"
        image.save(temp_file)

        # Faire les prédictions
        predicted_class = predict_image_class(temp_file, model, class_labels)

        # Supprimer le fichier temporaire
        os.remove(temp_file)

        st.write(f'Ce joueur est : {predicted_class}')  # Afficher la classe prédite sur l'interface

if __name__ == '__main__':
    main()
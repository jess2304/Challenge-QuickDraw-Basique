import keras
import numpy as np
from PIL import Image


def load_model_with_weights(model_name: str) -> keras.Model | None:
    """
    Importer le modèle à utiliser pour la prédiction

    Parameters
    ----------
    model_name: str
        Nom du fichier relatif au modèle

    Returns
    -------
    keras.Model
    """
    try:
        # Importer le modèle
        model = keras.models.load_model("src/models/" + model_name + ".h5")
        # Importer les poids liés au modèle
        model.load_weights("src/models/" + model_name + "_weights.h5")
        return model
    except OSError as e:
        # Erreur possible lorsque le fichier n'est pas trouvé ou est corrompu
        raise FileNotFoundError(
            f"Le fichier {model_name}.h5 ou {model_name}_weights.h5 est "
            "introuvable ou corrompu."
        ) from e
    except Exception as e:
        # Gestion d'autres exceptions générales liées au chargement du modèle
        raise RuntimeError(
            f"Une erreur inattendue est survenue lors du chargement du modèle: {str(e)}"
        ) from e


def preprocess_image(img: Image):
    """
    Réaliser un préprocessing de l'image pour qu'elle soit entrée au modèle

    Parameters
    ----------
    img: Image
        l'image en entrée

    Returns
    -------
    numpy.ndarray
        La matrice liée à l'image traitée
    """
    # Conertir en gris
    img = img.convert("L")
    # Convertir en tableau numpy et lui donner la taille qu'on veut
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(28, 28, 1)
    # Normaliser l'image
    img_array = img_array / 255.0
    # Ajouter la dimension du batch
    img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28, 1)

    return img_array

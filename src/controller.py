import keras


def load_model_with_weights(model_name: str) -> keras.Model:
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
        model = keras.models.load_model("models/" + model_name + ".h5")
        # Importer les poids liés au modèle
        model.load_weights("models/" + model_name + "_weights.h5")
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

"""
API pour appeler un modèle de classification QuickDraw
"""
import io

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

import controller as model_controller

# Créer l'API
api = FastAPI()

# Définir les classes
classes = ["basket", "eye", "binoculars", "rabbit", "hand"]


@api.post("/predict")
async def predict(file: UploadFile, model_name: str):
    """
    Prédire le résultat d'une image par le modèle donné

    Parameters
    ----------
    file: UploadFile
        Le fichier à prédire
    model_name: str
        Nom du modèle à utiliser pour prédire l'image
    """
    # Lire le fichier sur FastAPI
    img = await file.read()
    # Ouvrir l'image
    img = Image.open(io.BytesIO(img))
    # Réaliser le préprocessing de l'image
    img_array = model_controller.preprocess_image(img)
    # Importer le modèle
    model = model_controller.load_model_with_weights(model_name)
    if model is None:
        raise HTTPException(
            status_code=500, detail="Le modèle n'est pas chargé correctement."
        )
    # Prédire la classe de l'image
    predictions = model.predict(img_array)
    # Vérifier la forme des prédictions
    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError(f"Forme inattendue des prédictions: {predictions.shape}")
    # Obtenir l'indice de la classe avec la probabilité la plus élevée
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    # Vérifier que l'indice est dans les classes
    if predicted_class_idx < 0 or predicted_class_idx >= len(classes):
        raise ValueError(f"Indice de classe prédit hors limites: {predicted_class_idx}")
    # Mapper l'indice à un nom de classe
    predicted_class_name = classes[predicted_class_idx]
    # Retourner le nom de la classe prédite
    return {"prediction": predicted_class_name}


if __name__ == "__main__":
    uvicorn.run("api:api", host="127.0.0.1", port=8000, reload=False)

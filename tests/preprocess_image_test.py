# test preprocess image
import numpy as np
from PIL import Image

from controller import preprocess_image


def test_preprocess_image():
    # Créer une image d'exemple
    img = Image.new("RGB", (28, 28), color="white")
    # Appeler la fonction
    processed_img = preprocess_image(img)
    # Vérifier que l'image prétraitée a la bonne forme (1, 28, 28, 1)
    assert processed_img.shape == (
        1,
        28,
        28,
        1,
    ), "La forme de l'image prétraitée est incorrecte"
    # Vérifier que les valeurs sont normalisées entre 0 et 1
    assert np.all(processed_img >= 0) and np.all(
        processed_img <= 1
    ), "L'image n'est pas correctement normalisée"
    # Vérifier que l'image est bien en niveaux de gris
    assert np.allclose(
        processed_img, processed_img[:, :, :, 0:1]
    ), "L'image n'a pas été convertie en niveaux de gris"

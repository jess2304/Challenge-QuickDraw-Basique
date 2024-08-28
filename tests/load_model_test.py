# test_load_model.py
from unittest.mock import MagicMock, patch

import pytest

from controller import load_model_with_weights


def test_load_model_success():
    """
    Test unitaire pour importer le modèle avec succes
    """
    # Simuler un modèle keras
    mock_model = MagicMock()

    # Mock de load_model et load_weights
    with patch(
        "keras.models.load_model", return_value=mock_model
    ) as mock_load_model, patch.object(mock_model, "load_weights") as mock_load_weights:
        # Appeler la fonction à tester
        model = load_model_with_weights("test_model")

        # Vérifier que load_model et load_weights ont été appelés correctement
        mock_load_model.assert_called_once_with("models/test_model.h5")
        mock_load_weights.assert_called_once_with("models/test_model_weights.h5")

        # Vérifier que le modèle renvoyé est correct
        assert model == mock_model


def test_load_model_file_not_found():
    """
    Test unitaire pour importer le modèle mais fichier non trouvé
    """
    # Simuler l'exception OSError pour tester un fichier manquant
    with patch("keras.models.load_model", side_effect=OSError("File not found")):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_model_with_weights("test_model")

        # Vérifier le message d'erreur
        assert "test_model.h5 ou test_model_weights.h5 est introuvable" in str(
            excinfo.value
        )


def test_load_model_unexpected_error():
    """
    Test unitaire pour importer le modèle mais l'erreur est générale
    """
    # Simuler une erreur inattendue
    with patch("keras.models.load_model", side_effect=RuntimeError("Unexpected error")):
        with pytest.raises(RuntimeError) as excinfo:
            load_model_with_weights("test_model")
        # Vérifier le message d'erreur
        assert "Une erreur inattendue est survenue" in str(excinfo.value)

# CHALLENGE QuickDraw Basique + API avec FastAPI

## Introduction
Ce projet représente une API créée avec FastAPI pour lancer un modèle de classification qui donne la catégorie d'une image importée.
Les catégories sur lesquelles est entrainé le modèle sont : oeil, lunettes, lapin, main, panier
Les technologies utilisées sont : Tensorflow, Keras, FastAPI

## Etapes du projet
### Mise en place des Notebooks d'entraînement et de test
Ces notebooks Python ont été faits pendant mon cursus d'école d'ingénieur en Binôme avec mon camarade de classe Adama NANA, sous forme d'un examen de Deep Learning.
On a utilisé des techniques de Deep Learning pour créer le modèle, l'entraîner et le tester. Il reste toujours à améliorer bien sûr.

### Mise en place de l'API avec FastAPI
Cette partie concerne le lancement de l'application en utilisant le framework FastAPI qui est simple à utiliser.
Une approche basique a été faite concernant ce projet, vu que ce n'est pas une application très complexe.

## Comment lancer l'API
Il faut installer Python sur votre ordinateur pour démarrer l'API.
Dans le terminal vous pouvez créer et activer un nouvel environnement avec la commande :
> python -m venv venv
> venv\Scripts\activate
(venv est le nom de l'environnement)
Ensuite, pour l'installation des bibliothèques, il faut utiliser le fichier requirements.txt en tapant
> pip install -r requirements.txt
Enfin, il suffit de taper
> python .\src\api.py
Cela lancera l'API avec une génération d'URL que vous pourrez utiliser en ajoutant à la fin /docs pour accéder à l'IHM du Swagger FastAPI
https://fastapi.tiangolo.com/features/#automatic-docs

## Remarque
Le notebook challenge.ipynb contient les analyses et l'entraînement du modèle.
Le notebook de test permet de tester notre modèle.

# Présentation du projet
Ce dépôt contient le code source du projet IA réalisé dans le cadre du cursus ISI de l'ENSIMAG. Un site de démonstration du projet réalisé avec Streamlit est disponible [ici](https://ia-clothes.streamlit.app/).

## Organisation du dépôt
Le dépôt est organisé de la manière suivante :
```
.
├── data/
│   ├── datasets/
│   │   └── ...
│   ├── results/
│   │   ...
│   ├── Clustering.ipynb
│   └── Dataset_processing.ipynb
├── docs/
│   └── ...
└── frontend/
    └── ...
```
- `data/datasets/` contient des fichiers CSV décrivant les couleurs peau/haut/bas des images de nos datasets.
- `data/results/` contient les résultats de clustering et les matrices d'occurences pour différents paramétrages.
- `data/Clustering.ipynb` est le notebook contenant le code de clustering, disponible aussi [ici](https://colab.research.google.com/drive/10GGzqcxp0jIl4kTurNXCNJLFwRQY8UVE?usp=sharing) pour regénérer les résultats si souhaité.
- `data/Dataset_processing.ipynb` est le notebook contenant le code de détection des couleurs, disponible aussi [ici](https://colab.research.google.com/drive/19Hn6Y-09XlVNDg7Hp798v5ZL41UBsx8S?usp=sharing) pour regénérer les CSV de `data/datasets/` si souhaité.
- `docs/` contient les [livrables](#livrables) du projet.
- `frontend/` contient le code source du site web de démonstrations du projet.

## Livrables
Les livrables décrivant notre travail sont disponibles sur le site web du projet :
- [Livrable 1](https://nielk74.github.io/ia-clothes/#/livrable-1/)
- [Livrable 2](https://nielk74.github.io/ia-clothes/#/livrable-2/)
- [Livrable 3 (preuve de concept)](https://nielk74.github.io/ia-clothes/#/livrable-3/)
- [Livrable final](https://nielk74.github.io/ia-clothes/#/livrable-final/)

## Site de démonstration
Le site est déployable en local. Vous pouvez vous référer au README du dossier `frontend/` pour plus d'informations.
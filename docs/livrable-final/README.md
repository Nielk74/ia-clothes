# Livrable final

## Présentation du problème traité
Nous souhaitons proposer des idées de tenues vestimentaires en fonction de la personne. Pour simplifier le problème, nous avons décidé de nous concentrer sur l’association des couleurs des vêtements et de la couleur de la peau d’une personne. Nous cherchons à trouver les tuples de couleurs (peau, haut, bas) qui vont bien ensemble en se reposant sur des photos de mannequins. On peut ainsi suggérer des palettes de couleurs (haut, bas) à partir d’une teinte de peau.

## Données utilisées
Nous utilisons deux datasets pour les photos de mannequins :
- [DeepFashion](https://github.com/yumingj/DeepFashion-MultiModal) : Il contient 44 096 photos de mannequin dont beaucoup sont dupliquées avec des angles différents. Nous n'avons besoin que des photos où on voit le mannequin dans son ensemble (haut et bas) donc nous avons filtré le jeu données pour ne garder que les photos exploitables. Les images étant de relativement grande taille (environ 1000x1000), nous avons redimensionné les images afin d'accélérer le temps de traitement par la suite.

Script de filtrage basé sur le nom des fichiers :
```python
import glob
import os

images_path = "images/original"
trash_path = "images/trash"

original_files = glob.glob(images_path + "/*.jpg")

# for each image id, if there is a full type, keep only the full type
# otherwise, keep front and additional types because they can be fullbodies
for file in original_files:
    filename = file.split("/")[-1]
    similars = glob.glob(images_path + "/*" + filename.split("_")[2]+"*")
    if len(similars) >= 2:
        has_full = any("_full" in sim for sim in similars)
        if has_full:
            for similar in similars:
                if "_full" in similar:
                    continue
                else:
                    os.replace(similar, trash_path + "/" + similar.split("/")[-1])
        else:
            for similar in similars:
                if "_front" in similar or "_additional" in similar:
                    continue
                else:
                    os.replace(similar, trash_path + "/" + similar.split("/")[-1]) 
```

Script de redimensionnement des images :
```python
import cv2
import glob

images_path = "images/original"
resized_path = "images/resized"

original_files = glob.glob(images_path + "/*")

# resize images by 50%
for file in original_files:
    img = cv2.imread(file)
    img_50 = cv2.resize(img, None, fx = 0.50, fy = 0.50)
    cv2.imwrite(resized_path + "/" + file.split("/")[-1], img_50)
```
- [Style du Monde](https://styledumonde.com/) : TODO

Les jeux de données filtrés et redimensionnés sont disponibles [ici](https://drive.google.com/drive/folders/1_du47YFJGXp0veHWjdE59SLThpPCwxqg?usp=drive_link).

Ces jeux de données présentent tout de même plusieurs biais :
- Après filtrage, nous avons 1626 images d'hommes et 12 569 images de femmes dans le dataset DeepFashion.
- Les images proviennent d'une source occidentale donc toutes les populations et style ne sont pas représentées.
- Les images sont de qualité professionnelle avec une lumière permettant de bien percevoir les couleurs. Ce ne sera pas forcément le cas de photos prises par des utilisateurs donc le jeu de données ne représente pas parfaitement la réalité de notre cas d'usage.

Nous utilisons également un modèle pré-entraîné pour la détection de personnes et des habits : [Segformer](https://huggingface.co/mattmdjaga/segformer_b2_clothes). Ce modèle nous permet de distinguer le haut, le bas et la couleur de peau d’une personne.

## Méthodes utilisées et leur justification

## Évaluation des aspects environnementaux et sociétaux
L’outil que nous avons créé a un impact environnemental et sociétal tant sur le plan de la conception que sur celui de l’utilisation. 

### Impact de la conception
TODO
- cout pour la récupération des couleurs (temps, cpu, mémoire) : donner un ordre de grandeur de la quantité d'énergie utiisée
- cout pour le clsutering, la génération des matrices d'occurences

### Impact de l'utilisation
Le fait que notre application fasse des suggestions relatifs au style vestimentaire engendre des conséquences potentielles en matière de surconsommation de vêtements et donc de surconsommation de ressources et d'énergie. Il y a également le risque de normalisation des styles vestimentaires dictée par l'IA qui soulève des questions sur la diversité et l'individualité de l'expression personnelle.

Nous sommes également conscient que notre application porte sur les points sensibles que sont la couleur de peau et le traitement d'image personnelle. Dans le cadre de notre projet, des limites techniques et financières ont fait que nous avons déployé notre application sur un serveur tiers (Streamlit) et que nous avons hébergé les clusters et les matrices d'occurrences générés sur un dépôt GitHub public. Nous avons conscience que cela peut poser des problèmes de sécurité et de confidentialité. Nous avons donc réfléchi à des solutions pour pallier à ces problèmes :
- Flouter la photo qu'envoie l'utilisateur pour l'anonymiser
- Éviter d'utiliser un serveur tiers pour déployer notre application
- Chiffrer les échanges entre l'utilisateur et le serveur

## Bibliographie

## Code source
Le code source est disponible sur le dépôt GitHub du projet : https://github.com/Nielk74/ia-clothes
Pour comprendre sont contenu, il faut se référer au fichier README.md du dépôt.
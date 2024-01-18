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
- [Style du Monde](https://styledumonde.com/) : 7 841 photos ont constitué notre deuxième dataset. Ce site web partage des photographies de célébrités prises dans la rue ou lors d'évènements importants dans le domaine de la mode. L'intérêt était d'explorer des personnes pouvant porter des habits de tous les jours mais aussi plus originales. Nous avons scrapé toutes les photos proposées de Juillet 2008 à Septembre 2023 avec l'utilisation de *Scrapy* et retirer celles inexploitables (plusieurs personnes présentes, personne / visage non visible, pas une photo en pied...). Les photos ont été redimensionnées afin d'accélérer le temps de traitement par la suite.

Pour le scrapping, l'idée était de récupérer toutes les photos du site par année, en parcourant toutes les pages et les stocker dans un dossier. Le script ci-dessous a été lancé manuellement pour chaque année.

Script de scrapping :
```python
import scrapy
import urllib.request

class modelsSpider(scrapy.Spider): 
  name = "models"

  def start_requests(self):
    urls = [
      # "https://styledumonde.com/2008",
      # "https://styledumonde.com/2009",
      # "https://styledumonde.com/2010",
      "https://styledumonde.com/2011",
      # "... until 2023",
    ]
    # add the url of all the pages for the corresponding year
    for i in range (2, 4): # TODO: update the last index corresponding to the year
      urls.append("https://styledumonde.com/2011/page/"+str(i))
    for url in urls:
      yield scrapy.Request(url=url, callback=self.parse)

  def parse(self, response):
    delimiter = 'jpg'
    imgs = [img.attrib["src"] for img in response.css("img.attachment-original")]
    for img in imgs:
      urllib.request.urlretrieve(img, "scrapped_img/"+img.split("/")[-1].split(delimiter)[0]+delimiter)
```


Les jeux de données filtrés et redimensionnés sont disponibles [ici](https://drive.google.com/drive/folders/1_du47YFJGXp0veHWjdE59SLThpPCwxqg?usp=drive_link).

Ces jeux de données présentent tout de même plusieurs biais :
- Après filtrage, nous avons 1626 images d'hommes et 12 569 images de femmes dans le dataset DeepFashion.
- Les images proviennent d'une source occidentale donc toutes les populations et style ne sont pas représentées.
- Les images sont de qualité professionnelle avec une lumière permettant de bien percevoir les couleurs. Ce ne sera pas forcément le cas de photos prises par des utilisateurs donc le jeu de données ne représente pas parfaitement la réalité de notre cas d'usage.

Nous utilisons également un modèle pré-entraîné pour la détection de personnes et des habits : [Segformer](https://huggingface.co/mattmdjaga/segformer_b2_clothes). Ce modèle nous permet de distinguer le haut, le bas et la couleur de peau d’une personne.

## Méthodes utilisées et leur justification

## Évaluation des aspects environnementaux et sociétaux

## Bibliographie

## Code source
Le code source est disponible sur le dépôt GitHub du projet : https://github.com/Nielk74/ia-clothes
Pour comprendre sont contenu, il faut se référer au fichier README.md du dépôt.
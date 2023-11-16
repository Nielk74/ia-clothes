# Livrable 1
## La question étudiée
Nous souhaitons proposer des idées de tenues vestimentaires en fonction de la personne. Pour simplifier le problème, nous avons décidé de nous concentrer sur l’association des couleurs des vêtements et de la couleur de la peau d’une personne. Nous allons chercher à trouver les tuples de couleurs RGB (peau, haut, bas) qui vont bien ensemble en se reposant sur des photos de mannequins. On pourra alors suggérer des palettes de couleurs (haut, bas) recommandées à partir d’une teinte de peau. Nous pourrons éventuellement étendre notre projet vers de la suggestion de tenue si nous en avons le temps.

## Les données à utiliser
- Photos de mannequin : https://github.com/yumingj/DeepFashion-MultiModal

Ce dataset contient 44 096 photos de mannequin que nous pourrons exploiter pour extraire la couleur de peau du mannequin, la couleur du haut porté et la couleur du bas porté.

- Modèle pré-entraîné pour la détection de personnes et des habits : https://huggingface.co/mattmdjaga/segformer_b2_clothes 

Ce modèle nous permettra de distinguer le haut, le bas et la couleur de peau d’une personne.

- Dataset de tuples de couleurs (peau, haut, bas) à construire en analysant le dataset de mannequins.

## Aperçu des méthodes d’apprentissage à utiliser
Notre approche se découpe en deux étapes :
- Labellisation des couleurs présentes dans notre dataset de mannequins : on utilisera un algorithme de clustering pour découvrir des classes de couleur possible pour des vêtements et des teintes de peau. Ce processus pourra se faire en calculant la similarité entre les couleurs à partir d’une distance perceptuelle en utilisant le CIE76.
- Regroupement des photos par couleur de teinte, couleur de haut et couleur de bas en fonction du label associé
 
On proposera des palettes par intervalle de teinte de peau en fonction des occurrences les plus élevées dans notre regroupement.

## Réflexions préliminaires sur les enjeux environnementaux et sociétaux
L’outil que nous souhaitons créer a un impact environnemental et sociétal tant sur le plan de la conception que sur celui de l’utilisation. En ce qui concerne les coûts, nous devons évaluer les dépenses associées au fonctionnement de notre solution, susceptibles d'entraîner une augmentation de la consommation d'énergie. De même, le coût initial d'entraînement de l'IA sera à étudier en termes de ressources. De plus, il est nécessaire de prendre en compte le risque de surconsommation de vêtements, car nos suggestions pourraient encourager un cycle de renouvellement de la garde-robe plus rapide, intensifiant ainsi les problèmes inhérents à l'industrie de la mode. La normalisation des styles vestimentaires dictée par l'IA soulève des questions sur la diversité et l'individualité de l'expression personnelle, tandis que la dépendance à la technologie pourrait réduire la créativité individuelle.

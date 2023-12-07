import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colour
from colour.models import RGB_COLOURSPACE_sRGB
import csv


def colored_background(r, g, b, text):
    return f'\033[48;2;{r};{g};{b}m{text}\033[0m'

def get_data_from_input(path):
    dict_skin = {}

    with open(path, 'r' ) as infile:
        reader = csv.reader(infile, delimiter=';')
        # Iterate through each row in the input CSV and process the columns
        for row in reader:
            if 'None' in row[1] or 'None' in row[2] or 'None' in row[3]:
                continue 

            skin_column = row[3].strip()
            upper_column = row[2].strip()
            lower_column = row[1].strip()

            rgb_skin = np.array(skin_column.split(',')).astype(np.uint8)
            rgb_upper = np.array(upper_column.split(',')).astype(np.uint8)
            rgb_lower = np.array(lower_column.split(',')).astype(np.uint8)
            dict_skin[str(rgb_skin)] = {'upper':rgb_upper, 'lower':rgb_lower, 'skin': rgb_skin, 'path':row[0]}
    return dict_skin


def normalize_to_srgb(rgb):
    return np.dot(1/255,rgb)

# on construit un dicitionnaire de données de peau et de vêtements associés
# dict_skin = {'124, 123, 123': {'upper': [255, 255, 255], 'lower': [255, 255, 255], 'skin': [124, 123, 123], 'path': 'path/to/image'},...}
dict_skin = get_data_from_input('input.csv')
# dict_skin = get_data_from_input('input.csv') 

skin_data = []
for data in dict_skin.items():
    skin_data.append(data[1]['skin'])
# for rgb in skin_data:
#    print(colored_background(int(rgb[0][0]), int(rgb[0][1]), int(rgb[0][2]), f'{rgb[1]}'))
# exit(0)
skin_data = np.array(skin_data)
# Perform 3D clustering using KMeans
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
# lab_data = np.apply_along_axis(rgb_to_lab, 1, data)

tmp_skin_data = skin_data
# for i in range(len(skin_data)):
#     tmp_skin_data[i] = np.dot(m_lab, tmp_skin_data[i])
color_space = (RGB_COLOURSPACE_sRGB,)
lab_data = np.apply_along_axis(normalize_to_srgb, 1, skin_data)
lab_data = np.apply_along_axis(colour.RGB_to_XYZ, 1, lab_data, *color_space)
lab_data = np.apply_along_axis(colour.XYZ_to_Lab, 1, lab_data)

kmeans.fit(lab_data)
# Get cluster labels and cluster centers
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_


for i in range(num_clusters): # on parcourt les clusters de peau
    lab = cluster_centers[i] # on récupère le centre du cluster
    rgb = colour.Lab_to_XYZ([[lab[0], lab[1], lab[2]]]) # on convertit en RGB
    rgb = colour.XYZ_to_RGB(rgb, RGB_COLOURSPACE_sRGB)
    rgb = np.dot(255,rgb)
    len = skin_data[cluster_labels == i].shape[0] # on récupère le nombre de points dans le cluster
    # show color rgb
    print(colored_background(int(rgb[0][0]), int(rgb[0][1]), int(rgb[0][2]), f'Cluster {i + 1} --------------------------------------------------------------------------------------------------------- {len}'))
    cluster = skin_data[cluster_labels == i] # on récupère les points du cluster `i`
    upper_data = []
    lower_data = []
    for c in cluster:
        upper_data.append(dict_skin[str(c)]['upper']) # récupère les couleurs des vêtements associés dans le dictionnaire
        lower_data.append(dict_skin[str(c)]['lower'])
    upper_data = np.array(upper_data)
    lower_data = np.array(lower_data)
    # créations de clusters pour les données récupérées (vêtements)
    upper_lab_data = np.apply_along_axis(normalize_to_srgb, 1 , upper_data)
    lower_lab_data = np.apply_along_axis(normalize_to_srgb, 1, lower_data)
    upper_lab_data = np.apply_along_axis(colour.RGB_to_XYZ, 1, upper_lab_data, *color_space)
    lower_lab_data = np.apply_along_axis(colour.RGB_to_XYZ, 1, lower_lab_data, *color_space)
    upper_lab_data = np.apply_along_axis(colour.XYZ_to_Lab, 1, upper_lab_data)
    lower_lab_data = np.apply_along_axis(colour.XYZ_to_Lab, 1, lower_lab_data)
    n_colors = 15
    km_upper = KMeans(n_clusters=n_colors, n_init='auto')
    km_upper.fit(upper_lab_data)
    km_lower = KMeans(n_clusters=n_colors, n_init='auto')
    km_lower.fit(lower_lab_data)
    upper_cluster_labels = km_upper.labels_
    lower_cluster_labels = km_lower.labels_
    upper_cluster_centers = km_upper.cluster_centers_
    lower_cluster_centers = km_lower.cluster_centers_
    # get the top 3 color
    upper_color = []
    lower_color = []
    # on trie les clusters par nombre de points dans le cluster
    for j in range(n_colors):
        upper_color.append([upper_cluster_centers[j], upper_data[upper_cluster_labels == j].shape[0]])
        lower_color.append([lower_cluster_centers[j], lower_data[lower_cluster_labels == j].shape[0]])
    upper_color = sorted(upper_color, key=lambda x: x[1], reverse=True)
    lower_color = sorted(lower_color, key=lambda x: x[1], reverse=True)

    # on affiche les 5 premiers clusters pour les vêtements
    for c in upper_color[:]:
        lab = c[0]
        rgb = colour.Lab_to_XYZ([[lab[0], lab[1], lab[2]]])
        rgb = colour.XYZ_to_RGB(rgb, RGB_COLOURSPACE_sRGB)
        rgb = np.dot(255,rgb)
        # Filtrer couleur grise/noir/blanc
        diff = abs(rgb[0][0] - rgb[0][1]) + abs(rgb[0][1] - rgb[0][2]) + abs(rgb[0][0] - rgb[0][2])
        if diff > 15:
            print(colored_background(int(rgb[0][0]), int(rgb[0][1]), int(rgb[0][2]), f'Upper color ----------------------------- {c[1]} | {lab} | {diff}'))
    for c in lower_color[:]:
        lab = c[0]
        rgb = colour.Lab_to_XYZ([[lab[0], lab[1], lab[2]]])
        rgb = colour.XYZ_to_RGB(rgb, RGB_COLOURSPACE_sRGB)
        rgb = np.dot(255,rgb)
        diff = abs(rgb[0][0] - rgb[0][1]) + abs(rgb[0][1] - rgb[0][2]) + abs(rgb[0][0] - rgb[0][2])
        # Filtrer couleur grise/noir/blanc
        if diff > 15:
            print(colored_background(int(rgb[0][0]), int(rgb[0][1]), int(rgb[0][2]), f'Lower color ------------------------------ {c[1]} | {lab} | {diff}'))

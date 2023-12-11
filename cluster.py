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
            upper_column = row[1].strip()
            lower_column = row[2].strip()

            rgb_skin = np.array(skin_column.split(',')).astype(np.uint8)
            rgb_upper = np.array(upper_column.split(',')).astype(np.uint8)
            rgb_lower = np.array(lower_column.split(',')).astype(np.uint8)
            dict_skin[str(rgb_skin)] = {'upper':rgb_upper, 'lower':rgb_lower, 'skin': rgb_skin, 'path':row[0]}
    return dict_skin


def normalize_to_srgb(rgb):
    return np.dot(1/255,rgb)

def lab_to_rgb(lab):
    lab = np.array(lab)
    lab = np.reshape(lab, (1,3))
    lab = np.apply_along_axis(colour.Lab_to_XYZ, 1, lab)
    lab = np.apply_along_axis(colour.XYZ_to_RGB, 1, lab, *color_space)
    lab = np.apply_along_axis(np.dot, 1, lab, 255)
    return lab[0]

color_space = (RGB_COLOURSPACE_sRGB,)

def clustering(num_clusters, data):
    kmeans = KMeans(n_clusters=num_clusters)

    lab_data = np.apply_along_axis(normalize_to_srgb, 1, data)
    lab_data = np.apply_along_axis(colour.RGB_to_XYZ, 1, lab_data, *color_space)
    lab_data = np.apply_along_axis(colour.XYZ_to_Lab, 1, lab_data)

    kmeans.fit(lab_data)
    
    # Get cluster labels and cluster centers
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    return cluster_labels, cluster_centers


def get_occurences_by_skin_cluster(skin_cluster, occurences):
    occurences_by_skin_cluster = {}
    for key, value in occurences.items():
        if value['skin_cluster'] == skin_cluster:
            occurences_by_skin_cluster[key] = value
    occurences_by_skin_cluster = {k: v for k, v in sorted(occurences_by_skin_cluster.items(), key=lambda item: item[1]['occurences'], reverse=True)}
    return occurences_by_skin_cluster


# on construit un dicitionnaire de données de peau et de vêtements associés
# dict_skin = {'124, 123, 123': {'upper': [255, 255, 255], 'lower': [255, 255, 255], 'skin': [124, 123, 123], 'path': 'path/to/image'},...}
dict_skin = get_data_from_input('input.csv')
# dict_skin = get_data_from_input('input.csv') 

skin_data = []
upper_data =[]
lower_data = []

nb_cluster_skin = 5
nb_cluster_upper = 15
nb_cluster_lower = 15

for data in dict_skin.items():
    skin_data.append(data[1]['skin'])
    upper_data.append(data[1]['upper'])
    lower_data.append(data[1]['lower'])

skin_data = np.array(skin_data)
upper_data = np.array(upper_data)
lower_data = np.array(lower_data)

cluster_labels_skin, cluster_centers_skin = clustering(nb_cluster_skin, skin_data)
cluster_labels_upper, cluster_centers_upper = clustering(nb_cluster_upper, upper_data)
cluster_labels_lower, cluster_centers_lower = clustering(nb_cluster_lower, lower_data)

occurences = {}

for data in dict_skin.items():
    skin = data[1]['skin']
    upper = data[1]['upper']
    lower = data[1]['lower']
    path = data[1]['path']

    skin_cluster = cluster_labels_skin[np.where((skin_data == skin).all(axis=1))][0]
    upper_cluster = cluster_labels_upper[np.where((upper_data == upper).all(axis=1))][0]
    lower_cluster = cluster_labels_lower[np.where((lower_data == lower).all(axis=1))][0]
    
    # check if cluster_centers_upper[upper_cluster] and cluster_centers_lower[lower_cluster] are not too close
    rgb_upper = lab_to_rgb(cluster_centers_upper[upper_cluster])
    rgb_lower = lab_to_rgb(cluster_centers_lower[lower_cluster])
    diff_upper = abs(rgb_upper[0] - rgb_upper[1]) + abs(rgb_upper[1] - rgb_upper[2]) + abs(rgb_upper[0] - rgb_upper[2])
    diff_lower = abs(rgb_lower[0] - rgb_lower[1]) + abs(rgb_lower[1] - rgb_lower[2]) + abs(rgb_lower[0] - rgb_lower[2])
    key = str(skin_cluster) + str(upper_cluster) + str(lower_cluster)

    if key in occurences:
        occurences[key]['occurences'] += 1
    else:
        occurences[key] = {'skin_cluster': skin_cluster, 'upper_cluster': upper_cluster, 'lower_cluster': lower_cluster, 'occurences': 1, 'path': path}
    
# for each skin cluster, we print the 3 combinations of upper and lower clothes that are the most frequent
for skin_cluster in range(nb_cluster_skin):
    rgb_skin = lab_to_rgb(cluster_centers_skin[skin_cluster])
    print('\n')
    print(colored_background(int(rgb_skin[0]), int(rgb_skin[1]), int(rgb_skin[2]), f'Skin {skin_cluster}'), end=' ')
    print('\n')
    occurences_by_skin_cluster = get_occurences_by_skin_cluster(skin_cluster, occurences)
    for i in range(5):
        key = list(occurences_by_skin_cluster.keys())[i]
        value = occurences_by_skin_cluster[key]
        rgb_upper = lab_to_rgb(cluster_centers_upper[value['upper_cluster']])
        rgb_lower = lab_to_rgb(cluster_centers_lower[value['lower_cluster']])
        print("Occurences : " + str(value['occurences']), end=' ')
        print(colored_background(int(rgb_upper[0]), int(rgb_upper[1]), int(rgb_upper[2]), f'Upper {value["upper_cluster"]}'), end=' ')
        print(colored_background(int(rgb_lower[0]), int(rgb_lower[1]), int(rgb_lower[2]), f'Lower {value["lower_cluster"]}'), end=' ')
        print("Exemple d'image : " + value['path'])
        print("\n")
        # print('\t'+colored_background(int(cluster_centers_upper[value['upper_cluster']][0]*255), int(cluster_centers_upper[value['upper_cluster']][1]*255), int(cluster_centers_upper[value['upper_cluster']][2]*255), f'Upper {value["upper_cluster"]}'), end=' ')
        # print(colored_background(int(cluster_centers_lower[value['lower_cluster']][0]*255), int(cluster_centers_lower[value['lower_cluster']][1]*255), int(cluster_centers_lower[value['lower_cluster']][2]*255), f'Lower {value["lower_cluster"]}'), end=' ')
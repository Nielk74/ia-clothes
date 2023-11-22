import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage
import colour
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



dict_skin = get_data_from_input('input.csv')
skin_data = []
for data in dict_skin.items():
    skin_data.append([data[1]['upper'],data[1]['path']])
for rgb in skin_data:
   print(colored_background(int(rgb[0][0]), int(rgb[0][1]), int(rgb[0][2]), f'{rgb[1]}'))
exit(0)
skin_data = np.array(skin_data)
# Perform 3D clustering using KMeans
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters)
def distance(a, b):
    return 0
# lab_data = np.apply_along_axis(rgb_to_lab, 1, data)
lab_data = np.apply_along_axis(colour.XYZ_to_Lab, 1, skin_data)
kmeans.eucledian_distances = distance
kmeans.fit(lab_data)
# Get cluster labels and cluster centers
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_




# Visualize the clustered points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot points with their assigned clusters
for i in range(num_clusters):
    # show color of each center
    if 1==0:
        rgb = cluster_centers[i]
        print(colored_background(int(rgb[0]), int(rgb[1]), int(rgb[2]), f'Cluster {i + 1}'))
    else:
        lab = cluster_centers[i]
        rgb = colour.Lab_to_XYZ([[lab[0], lab[1], lab[2]]])
        len = skin_data[cluster_labels == i].shape[0]
        # show color rgb
        print(colored_background(int(rgb[0][0]), int(rgb[0][1]), int(rgb[0][2]), f'Cluster {i + 1} : {len}'))

for i in range(num_clusters):
    rgb = [skin_data[cluster_labels == i][0][0]/255, skin_data[cluster_labels == i][0][1]/255, skin_data[cluster_labels == i][0][2]/255]
    ax.scatter(skin_data[cluster_labels == i, 0], skin_data[cluster_labels == i, 1], skin_data[cluster_labels == i, 2], label=f'Cluster {i + 1}', c=[rgb])

# for c in data:
#     rgb = [c[0]/255, c[1]/255, c[2]/255]
#     ax.scatter(c[0], c[1], c[2], label=f'Cluster {i + 1}', c=[rgb])
# Plot cluster centers
# for i in range(num_clusters):
#     center_in_lab = colour.XYZ_to_Lab([[cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2]]])
#     ax.scatter(center_in_lab[0][0]*255, center_in_lab[0][1]*255, center_in_lab[0][2]*255, s=100, c='black', marker='x')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# for c in data:
#     print(colored_background(int(c[0]), int(c[1]), int(c[2]), f'---------------'))
# Add legend
ax.legend()

# Show the 3D plot
plt.show()
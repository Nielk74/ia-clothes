import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage
import colour



def colored_background(r, g, b, text):
    return f'\033[48;2;{r};{g};{b}m{text}\033[0m'


data = np.loadtxt('output.csv', delimiter=',')  # Replace 'your_file_path.txt' with the actual file path

# Perform 3D clustering using KMeans
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters)
def distance(a, b):
    return 0
# lab_data = np.apply_along_axis(rgb_to_lab, 1, data)
lab_data = np.apply_along_axis(colour.XYZ_to_Lab, 1, data)
kmeans.eucledian_distances = distance
kmeans.fit(data)
# Get cluster labels and cluster centers
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Visualize the clustered points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot points with their assigned clusters
for i in range(num_clusters):
    # show color of each center
    if 1==1:
        rgb = cluster_centers[i]
        len = data[cluster_labels == i].shape[0]

        print(colored_background(int(rgb[0]), int(rgb[1]), int(rgb[2]), f'Cluster {i + 1} : {len}'))
    else:
        lab = cluster_centers[i]
        rgb = colour.Lab_to_XYZ([[lab[0], lab[1], lab[2]]])
        len = data[cluster_labels == i].shape[0]
        # show color rgb
        print(colored_background(int(rgb[0][0]), int(rgb[0][1]), int(rgb[0][2]), f'Cluster {i + 1} : {len}'))

# for i in range(num_clusters):
#     rgb = [data[cluster_labels == i][0][0]/255, data[cluster_labels == i][0][1]/255, data[cluster_labels == i][0][2]/255]
#     ax.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], data[cluster_labels == i, 2], label=f'Cluster {i + 1}', c=[rgb])

for i in range(num_clusters):
    rgb = [data[cluster_labels == i][0][0]/255, data[cluster_labels == i][0][1]/255, data[cluster_labels == i][0][2]/255]
    ax.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], data[cluster_labels == i, 2], label=f'Cluster {i + 1}', c=[rgb])


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# for c in data:
#     print(colored_background(int(c[0]), int(c[1]), int(c[2]), f'---------------'))
# Add legend
ax.legend()

# Show the 3D plot
plt.show()
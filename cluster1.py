import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage
import math

def colored_background(r, g, b, text):
    return f'\033[48;2;{r};{g};{b}m{text}\033[0m'
def hsv_to_rgb(hsv):
    """
    Convert HSV to RGB color space
    """
    h, s, v = hsv
    h = h / 360.0
    s = s / 100.0
    v = v / 100.0
    if s == 0.0:
        v *= 255
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = int(255 * v * (1.0 - s))
    q = int(255 * v * (1.0 - (s * f)))
    t = int(255 * v * (1.0 - (s * (1.0 - f))))
    v = int(v * 255)
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
    


def rgb_to_hsv(rgb):
    """
    Convert RGB to HSV color space
    :param rgb: RGB color
    :return: HSV color (tuple)
    """
    # unpack
    r, g, b = rgb
    # normalize
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    # convert
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df / mx) * 100
    v = mx * 100
    return math.sin(math.pi*h/360), s, v
data = np.loadtxt('output.csv', delimiter=',')  # Replace 'your_file_path.txt' with the actual file path

num_clusters = 100
kmeans = KMeans(n_clusters=num_clusters)
lab_data = np.apply_along_axis(rgb_to_hsv, 1, data)
kmeans.fit(lab_data)

# Get cluster labels and cluster centers
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Visualize the clustered points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot points with their assigned clusters
max = 0
cmax = ""
for i in range(num_clusters):
    # count the number of points in each cluster
    print(f'Cluster {i+1}: {np.count_nonzero(cluster_labels == i)} points')
    if max < np.count_nonzero(cluster_labels == i):
        max = np.count_nonzero(cluster_labels == i)
        hsv = cluster_centers[i]
        print(hsv)
        rgb = hsv_to_rgb([hsv[0], hsv[1], hsv[2]])
        cmax = rgb
    # show color of each center
    if 1==1:
        hsv = cluster_centers[i]
        print(hsv)
        rgb = hsv_to_rgb([hsv[0], hsv[1], hsv[2]])
        print(colored_background(int(rgb[0]), int(rgb[1]), int(rgb[2]), f'Cluster {i + 1}'))
    else:
        lab = cluster_centers[i]
        rgb = skimage.color.lab2rgb([[lab[0], lab[1], lab[2]]])
        # show color rgb
        print(colored_background(int(rgb[0][0]*255), int(rgb[0][1]*255), int(rgb[0][2]*255), f'Cluster {i + 1}'))

print(colored_background(int(cmax[0]), int(cmax[1]), int(cmax[2]), f'max : {max}'))
for i in range(num_clusters):
    # plot as a hsv color
    hsv = cluster_centers[i]
    rgb = hsv_to_rgb([hsv[0], hsv[1], hsv[2]])
    rgb = [rgb[0]/255, rgb[1]/255, rgb[2]/255]
    ax.scatter(lab_data[cluster_labels == i, 0], lab_data[cluster_labels == i, 1], lab_data[cluster_labels == i, 2], label=f'Cluster {i + 1}', c=[rgb])

# Plot cluster centers
for i in range(num_clusters):
    # plot as a hsv color
    hsv = cluster_centers[i]
    rgb = hsv_to_rgb([hsv[0], hsv[1], hsv[2]])
    rgb = [rgb[0]/255, rgb[1]/255, rgb[2]/255]
    ax.scatter(cluster_centers[i, 0], cluster_centers[i, 1], cluster_centers[i, 2], c=rgb, marker='X', s=200, label='Cluster Centers')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

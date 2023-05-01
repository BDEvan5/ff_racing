import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# Example lidar scan data (distances from car to points on the track)
# lidar_data = np.random.rand(1080) * 50

lidar_data = np.load("Data/MyFrenetPlanner/ScanData/MyFrenetPlanner_0.npy")

# Cluster the lidar data using DBSCAN
clusterer = DBSCAN(eps=1, min_samples=10)
clusters = clusterer.fit_predict(lidar_data.reshape(-1, 1))

# Fit a line through the center of each cluster to extract the center line of the track
center_line = []
for cluster_label in np.unique(clusters):
    if cluster_label == -1:  # ignore noise points
        continue
    cluster_mask = (clusters == cluster_label)
    cluster_data = lidar_data[cluster_mask]
    cluster_center = np.mean(cluster_data)
    center_line.append(cluster_center)

plt.figure()

fov2 = 4.7 / 2
angles = np.linspace(-fov2, fov2, 1080)
coses = np.cos(angles)
sines = np.sin(angles)
plt.plot(lidar_data*coses, lidar_data*sines)

print(center_line)
plt.plot(center_line)

plt.show()

# Fit a line through the extracted center line points
x = np.arange(len(center_line)).reshape(-1, 1)
regressor = LinearRegression().fit(x, center_line)
slope = regressor.coef_[0]
intercept = regressor.intercept_

print("Center line slope:", slope)
print("Center line intercept:", intercept)

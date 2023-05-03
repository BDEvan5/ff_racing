import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


lidar_data = np.load("Data/MyFrenetPlanner/ScanData/MyFrenetPlanner_0.npy")

plt.figure()
fov2 = 4.7 / 2
angles = np.linspace(-fov2, fov2, 1080)
coses = np.cos(angles)
sines = np.sin(angles)
n2 = 540

xs = coses * lidar_data
ys = sines * lidar_data

c_xs = (xs[:n2] + xs[n2:][::-1])/2
c_ys = (ys[:n2] + ys[n2:][::-1])/2

plt.plot(xs, ys, label="Lidar")
plt.plot(c_xs, c_ys, label="Center")

plt.show()
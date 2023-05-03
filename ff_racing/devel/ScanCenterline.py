import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

def basic_centerline():
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


def end_pt_line():
    lidar_data = np.load("Data/MyFrenetPlanner/ScanData/MyFrenetPlanner_0.npy")

    plt.figure()
    fov2 = 4.7 / 2
    angles = np.linspace(-fov2, fov2, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)
    n2 = 540

    xs = coses * lidar_data
    ys = sines * lidar_data

    pts = np.hstack((xs[:, None], ys[:, None]))
    dists = np.linalg.norm(np.diff(pts, axis=0))
    min_idx = np.argmin(dists)

    c_xs = (xs[:n2] + xs[n2:][::-1])/2
    c_ys = (ys[:n2] + ys[n2:][::-1])/2

    plt.plot(xs, ys, label="Lidar")
    plt.plot(c_xs, c_ys, label="Center")

    plt.plot(pts[min_idx:min_idx+2, 0], pts[min_idx:min_idx+2, 1], 'red')

    plt.show()

end_pt_line()

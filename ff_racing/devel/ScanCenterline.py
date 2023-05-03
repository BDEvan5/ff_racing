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
    pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    min_idx = np.argmax(pt_distances)

    c_xs = (xs[:n2] + xs[n2:][::-1])/2
    c_ys = (ys[:n2] + ys[n2:][::-1])/2

    plt.plot(xs, ys, 'x-', label="Lidar")
    plt.plot(c_xs, c_ys, label="Center")

    plt.plot(pts[min_idx:min_idx+2, 0], pts[min_idx:min_idx+2, 1], 'red')
    
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()
    
def true_centerline():
    lidar_data = np.load("Data/MyFrenetPlanner/ScanData/MyFrenetPlanner_0.npy")

    plt.figure()
    fov2 = 4.7 / 2
    angles = np.linspace(-fov2, fov2, 1080)
    coses = np.cos(angles)
    sines = np.sin(angles)

    xs = coses * lidar_data
    ys = sines * lidar_data

    pts = np.hstack((xs[:, None], ys[:, None]))
    pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    mid_idx = np.argmax(pt_distances)

    length_1 = np.sum(pt_distances[:mid_idx])
    length_2 = np.sum(pt_distances[mid_idx:])
    
    l1_ss = np.linspace(0, length_1, 50)
    l1_ss = np.linspace(0, length_2, 50)
    
    l1_cs = np.cumsum(pt_distances[:mid_idx+1])
    l2_cs = np.cumsum(pt_distances[mid_idx:])

    l1_xs = np.interp(l1_ss, l1_cs, pts[:mid_idx+1, 0])
    l1_ys = np.interp(l1_ss, l1_cs, pts[:mid_idx+1, 1])
    l2_xs = np.interp(l1_ss, l2_cs, pts[mid_idx+1:, 0])
    l2_ys = np.interp(l1_ss, l2_cs, pts[mid_idx+1:, 1])


    c_xs = (l1_xs + l2_xs[::-1])/2
    c_ys = (l1_ys + l2_ys[::-1])/2

    plt.plot(xs, ys, 'x', label="Lidar")
    
    plt.plot(l1_xs, l1_ys, label="Center - 1")
    plt.plot(l2_xs, l2_ys, label="Center - 2")
    
    plt.plot(c_xs, c_ys, '-', color='red', label="Center", linewidth=3)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend()
    plt.show()

# end_pt_line()
true_centerline()
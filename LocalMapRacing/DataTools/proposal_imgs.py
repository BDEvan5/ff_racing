import numpy as np 
from matplotlib import pyplot as plt
from LocalMapRacing.local_mapping.LocalMap import LocalMap, PlotLocalMap
from LocalMapRacing.DataTools.MapData import MapData
from LocalMapRacing.DataTools.plotting_utils import *

from LocalMapRacing.planner_utils.utils import ensure_path_exists
from matplotlib.patches import RegularPolygon
from matplotlib.patches import Polygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def make_img(name, i):
    path  = f"Data/{name}/"
    scan_root = path + "ScanData/"
    map_name = "aut"
    img_path = path + "Imgs/"

    history = np.load(path + "TestingAUT/" + f"Lap_0_history_{name}.npy")
    states = history[:, 0:7]
    actions = history[:, 7:9]

    position = states[i, 0:2]
    heading = states[i, 4]


    map_data = MapData(map_name)
    origin = map_data.map_origin[:2]

    file = scan_root + f"scan_{i}.npy"
    scan = np.load(file)

    # plot scan in eng
    plt.figure(1)
    plt.clf()
    
    map_data.plot_map_img()
    x, y = map_data.xy2rc(states[i, 0], states[i, 1])

    angles = np.linspace(-4.7/2, 4.7/2, 1080)
    sines = np.sin(angles)
    cosines = np.cos(angles)
    xs, ys = cosines * scan, sines * scan
    scan_pts = np.column_stack((xs, ys))

    rotation = np.array([[np.cos(heading), -np.sin(heading)],
                                [np.sin(heading), np.cos(heading)]])


    scan_pts = np.matmul(rotation, scan_pts.T).T
    scan_pts = scan_pts + position
    scan_pts = (scan_pts - origin) / map_data.map_resolution

    x, y = map_data.xy2rc(states[i, 0], states[i, 1])
    plt.plot(x, y, 'x', color='red')

    img = plt.imread("LocalMapRacing/DataTools/RacingCar.png", format='png')
    img = rotate_bound(img, 150)
    oi = OffsetImage(img, zoom=0.5)
    ab = AnnotationBbox(oi, (x-3.5, y+6), xycoords='data', frameon=False)
    plt.gca().add_artist(ab)

    plt.plot(scan_pts[:, 0], scan_pts[:, 1], 'x', color=color_pallette[0])

    poly_pts = np.array([[x, y], scan_pts[0], scan_pts[100], scan_pts[452], scan_pts[453], scan_pts[480], scan_pts[500], scan_pts[530], scan_pts[550], scan_pts[570], scan_pts[590], scan_pts[650], scan_pts[-1]])
    poly = Polygon(poly_pts, color=color_pallette[0], alpha=0.3)
    plt.gca().add_patch(poly)

    plt.axis('equal')
    plt.gca().axis('off')

    b = 10
    plt.xlim([np.min(scan_pts[:, 0]) - b, np.max(scan_pts[:, 0]) + b])
    plt.ylim([np.min(scan_pts[:, 1]) - b, np.max(scan_pts[:, 1]) + b])

    plt.tight_layout()

    plt.savefig(img_path + f"env_scan_{i}.svg", bbox_inches='tight', pad_inches=0)
    return
    plt.show()

    # plot local map

    local_track = np.load(file)
    local_map = PlotLocalMap(local_track)

    local_map.plot_local_map(img_path, i)

    plt.figure(5)
    plt.clf()
    map_data.plot_map_img()
    x, y = map_data.xy2rc(states[i, 0], states[i, 1])
    plt.plot(x, y, 'x', color='red')

    plt.gca().axis('off')

    plt.savefig(img_path + f"local_pos_{i}.svg", bbox_inches='tight', pad_inches=0)


    local_map.plot_local_map_offset(position, heading, map_data.map_origin[:2], map_data.map_resolution, full_path, i)

    # plot traj

def rotate_bound(image, angle):
    import cv2
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255,255,255))


make_img("LocalImgs", 91)
# plot_local_maps("Data/devel_local_mpcc/")







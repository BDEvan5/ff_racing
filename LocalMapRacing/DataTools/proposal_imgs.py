import numpy as np 
from matplotlib import pyplot as plt
from LocalMapRacing.local_mapping.LocalMap import LocalMap, PlotLocalMap
from LocalMapRacing.local_mapping.LocalMapGenerator import LocalMapGenerator
from LocalMapRacing.local_mapping.LocalRaceline import LocalRaceline
from LocalMapRacing.DataTools.MapData import MapData
from LocalMapRacing.DataTools.plotting_utils import *

from LocalMapRacing.planner_utils.utils import ensure_path_exists
from matplotlib.patches import RegularPolygon
from matplotlib.patches import Polygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import LineCollection

text_size = 14
text_size2 = 20

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
    plt.figure(1, figsize=(3, 6))
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

    plt.plot(scan_pts[:, 0], scan_pts[:, 1], 'o', color=color_pallette[0], label="LiDAR Scan")

    poly_pts = np.array([[x, y], scan_pts[0], scan_pts[100], scan_pts[452], scan_pts[453], scan_pts[480], scan_pts[500], scan_pts[530], scan_pts[550], scan_pts[570], scan_pts[590], scan_pts[650], scan_pts[-1]])
    poly = Polygon(poly_pts, color=color_pallette[0], alpha=0.3)
    plt.gca().add_patch(poly)

    plt.gca().axis('off')
    b = 6
    plt.xlim([np.min(scan_pts[:, 0]) - b, np.max(scan_pts[:, 0]) + b])
    plt.ylim([np.min(scan_pts[:, 1]) - 16, np.max(scan_pts[:, 1]) + b])
    plt.text(x - 25, y - 73, "Track", fontsize=text_size2)

    # plt.legend(loc='upper right', fontsize=text_size)
    plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True)

    plt.tight_layout()

    plt.savefig(img_path + f"env_scan_{i}.svg", bbox_inches='tight', pad_inches=0)

    # plot local map
    plt.clf()
    lm_generator = LocalMapGenerator("Data/LocalImgs/")

    xs, ys = cosines * scan, sines * scan
    pts, pt_distances, inds = lm_generator.extract_full_track_lines(xs, ys)
    long_side, short_side = lm_generator.extract_boundaries(pts, pt_distances, inds)
    # track = lm_generator.project_side_to_track(long_side, w, n_pts)
    # local_map = lm_generator.adjust_track_normals(track)

    # lm = lm_generator.generate_line_local_map(scan, False)
    # lm.plot_local_map(img_path, i)
    map_data.plot_map_img()

    long_side = (np.matmul(rotation, long_side.T).T + position - origin ) / map_data.map_resolution
    short_side = (np.matmul(rotation, short_side.T).T + position - origin ) / map_data.map_resolution

    plt.plot(scan_pts[:, 0], scan_pts[:, 1], 'o', color=color_pallette[0], alpha=0.9)
    # plt.plot(scan_pts[:, 0], scan_pts[:, 1], 'x', color=color_pallette[0], label="Scan", alpha=0.9)
    boundary_color = color_pallette[2]
    plt.plot(long_side[:, 0], long_side[:, 1], '-', color=boundary_color, linewidth=3)
    plt.plot(short_side[:, 0], short_side[:, 1], '-', color=boundary_color, label="Boundaries", linewidth=3)

    img = plt.imread("LocalMapRacing/DataTools/RacingCar.png", format='png')
    img = rotate_bound(img, 150)
    oi = OffsetImage(img, zoom=0.5)
    ab = AnnotationBbox(oi, (x-3.5, y+6), xycoords='data', frameon=False)
    plt.gca().add_artist(ab)

    poly_pts = np.array([[x, y], scan_pts[0], scan_pts[100], scan_pts[452], scan_pts[453], scan_pts[480], scan_pts[500], scan_pts[530], scan_pts[550], scan_pts[570], scan_pts[590], scan_pts[650], scan_pts[-1]])
    poly = Polygon(poly_pts, color=color_pallette[0], alpha=0.2)
    plt.gca().add_patch(poly)

    plt.gca().axis('off')
    b = 8
    plt.xlim([np.min(short_side[:, 0]) - b, np.max(long_side[:, 0]) + b])
    plt.ylim([np.min(long_side[:, 1]) - 18, np.max(long_side[:, 1]) + b])

    plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.5, 0.055))
    # plt.legend(loc='upper right', fontsize=12)

    plt.savefig(img_path + f"env_boundaries_{i}.svg", bbox_inches='tight', pad_inches=0)

    # plot local map
    plt.clf()
    # local_map = lm_generator.adjust_track_normals(track)

    boundary_color = 'black'
    # boundary_color = color_pallette[2]
    plt.plot(long_side[:, 0], long_side[:, 1], '-', color=boundary_color, linewidth=2, alpha=1)
    plt.plot(short_side[:, 0], short_side[:, 1], '-', color=boundary_color, linewidth=2, alpha=1)
    # plt.plot(short_side[:, 0], short_side[:, 1], '-', color=boundary_color, label="Edges", linewidth=2, alpha=0.7)

    img = plt.imread("LocalMapRacing/DataTools/RacingCar.png", format='png')
    img = rotate_bound(img, 150)
    oi = OffsetImage(img, zoom=0.5)
    ab = AnnotationBbox(oi, (x-3.5, y+6), xycoords='data', frameon=False)
    plt.gca().add_artist(ab)

    long_side_xy, short_side_xy = lm_generator.extract_boundaries(pts, pt_distances, inds)
    track = lm_generator.project_side_to_track(long_side_xy, -1, int(len(long_side_xy) / 2))
    # track_long = lm_generator.project_side_to_track(long_side_xy, -1, int(len(long_side_xy) / 2))
    # track_short = lm_generator.project_side_to_track(short_side_xy, 1, int(len(short_side_xy) / 2))

    #TODO: find a method to combine the two tracks

    # adj_lm = lm_generator.adjust_track_normals(track)
    # track = adj_lm.track
    track_pts = (np.matmul(rotation, track[:, :2].T).T + position - origin ) / map_data.map_resolution
    plt.plot(track_pts[2:, 0], track_pts[2:, 1], '-', color=color_pallette[3], label="Centre line", linewidth=3)
    # plt.plot(track_pts[2:, 0], track_pts[2:, 1], '-', color=color_pallette[3], label="Centre\nline", linewidth=3)

    #plot nvecs
    lm = LocalMap(track)
    l1 = lm.track[:, :2] + lm.nvecs * lm.track[:, 2][:, None]
    l2 = lm.track[:, :2] - lm.nvecs * lm.track[:, 3][:, None]

    l1 = (np.matmul(rotation, l1.T).T + position - origin ) / map_data.map_resolution
    l2 = (np.matmul(rotation, l2.T).T + position - origin ) / map_data.map_resolution

    for z in range(3, len(l1)):
        n_xs = [l1[z, 0], l2[z, 0]]
        n_ys = [l1[z, 1], l2[z, 1]]
        # plt.plot(n_xs, n_ys, '-', color=science_pallet[3], linewidth=2)
        plt.plot(n_xs, n_ys, '-', color=science_bright[0], linewidth=2)
    
    n_xs = [l1[2, 0], l2[2, 0]]
    n_ys = [l1[2, 1], l2[2, 1]]
    # plt.plot(n_xs, n_ys, '-', color=science_bright[0], linewidth=2, label="Normal\nVectors")
    plt.plot(n_xs, n_ys, '-', color=science_bright[0], linewidth=2, label="Normal Vectors")

    map_data = MapData(map_name)
    map_data.plot_map_img_light()
    
    plt.gca().axis('off')
    b = 6
    plt.xlim([np.min(short_side[:, 0]) - b, np.max(long_side[:, 0]) + 28])
    # plt.xlim([np.min(short_side[:, 0]) - 20, np.max(long_side[:, 0]) + 8])
    plt.ylim([np.min(long_side[:, 1]) - b, np.max(long_side[:, 1]) + b])
    plt.gca().set_aspect('equal')

    # plt.legend(loc='center', fontsize=12, bbox_to_anchor=(0.25, 0.2))
    # plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.5, 0.05), ncol=2)
    plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.5, -0.07), ncol=1)
    # plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.75, 0.9))
    # plt.legend(loc='upper right', fontsize=12)

    plt.savefig(img_path + f"local_map_{i}.svg", bbox_inches='tight', pad_inches=0)


    # plot traj

    plt.clf()
    boundary_color = 'black'
    # boundary_color = color_pallette[2]
    plt.plot(long_side[:, 0], long_side[:, 1], '-', color=boundary_color, linewidth=2, alpha=1)
    plt.plot(short_side[:, 0], short_side[:, 1], '-', color=boundary_color, label="Boundaries", linewidth=2, alpha=1)

    img = plt.imread("LocalMapRacing/DataTools/RacingCar.png", format='png')
    img = rotate_bound(img, 150)
    oi = OffsetImage(img, zoom=0.5)
    ab = AnnotationBbox(oi, (x-3.5, y+6), xycoords='data', frameon=False)
    plt.gca().add_artist(ab)

    long_side_xy, short_side_xy = lm_generator.extract_boundaries(pts, pt_distances, inds)
    track = lm_generator.project_side_to_track(long_side_xy, -1, int(len(long_side_xy) / 2))
    track_pts = (np.matmul(rotation, track[:, :2].T).T + position - origin ) / map_data.map_resolution
    plt.plot(track_pts[:, 0], track_pts[:, 1], '--', color='black', label="Centre line", linewidth=2.5)
    # plt.plot(track_pts[:, 0], track_pts[:, 1], '-', color=color_pallette[3], label="Centre line", linewidth=2)

    #plot nvecs
    # adj_lm = lm_generator.adjust_track_normals(track)
    # track = adj_lm.track
    lm = LocalMap(track[2:])
    lr = LocalRaceline("Data/LocalImgs/")
    lr.generate_raceline(lm)

    points = (np.matmul(rotation, lr.raceline.T).T + position - origin ) / map_data.map_resolution
    points = points.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # print(lr.vs)
    norm = plt.Normalize(2, 8)
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(lr.vs)
    lc.set_linewidth(5)
    line = plt.gca().add_collection(lc)
    cbar = plt.colorbar(line, shrink=0.5)
    # cbar = plt.colorbar(line, label="Speed")
    cbar.ax.tick_params(labelsize=15)
    # cbar.ax.set_yticks([])
    cbar.ax.set_yticklabels(["", "", "", "", "", ' '], fontsize=15)
    # cbar.ax.set_yticklabels(['Slow', "", "", "", "", "", 'Fast'], fontsize=15)
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.text(0.78, 0.25, "Slow       >>>>        Fast", fontsize=15, transform=plt.gcf().transFigure, rotation=90)
    plt.text(0.74, 0.45, "Speed", fontsize=text_size, transform=plt.gcf().transFigure, rotation=90)

    map_data = MapData(map_name)
    map_data.plot_map_img_light()

    plt.gca().axis('off')
    b = 6
    plt.xlim([np.min(short_side[:, 0]) - b, np.max(long_side[:, 0]) + 4])
    plt.ylim([np.min(long_side[:, 1]) - b, np.max(long_side[:, 1]) + b])
    plt.gca().set_aspect('equal')

    # plt.legend(loc='center', fontsize=12, bbox_to_anchor=(0.75, 0.9))

    plt.savefig(img_path + f"local_trajectory_{i}.svg", bbox_inches='tight', pad_inches=0)


def rotate_bound(image, angle):
    import cv2
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderValue=(1, 1, 1))


make_img("LocalImgs", 91)
# plot_local_maps("Data/devel_local_mpcc/")







import numpy as np 
from matplotlib import pyplot as plt
from LocalMapRacing.local_mapping.LocalMap import LocalMap, PlotLocalMap
from LocalMapRacing.DataTools.MapData import MapData

from LocalMapRacing.planner_utils.utils import ensure_path_exists

def plot_local_maps(name):
    path  = f"Data/{name}/"
    map_root = path + "LocalMapData/"
    map_name = "aut"
    lm_path = path + "LocalMapImgs/"
    pos_path = path + "LocalMapPos/"
    full_path = path + "LocalMapFull/"
    ensure_path_exists(lm_path)
    ensure_path_exists(pos_path)
    ensure_path_exists(full_path)

    history = np.load(path + "TestingAUT/" + f"Lap_0_history_{name}.npy")
    states = history[:, 0:7]
    actions = history[:, 7:9]

    map_data = MapData(map_name)

    # plt.plot(states[:, 4])
    # plt.show()

    track_lengths = []
    for i in range(1, 260):
        file = map_root + f"local_map_{i}.npy"
        try:
            local_track = np.load(file)
        except: break
        local_map = PlotLocalMap(local_track)

        local_map.plot_local_map(lm_path, i)

        plt.figure(5)
        plt.clf()
        map_data.plot_map_img()
        x, y = map_data.xy2rc(states[i, 0], states[i, 1])
        plt.plot(x, y, 'x', color='red')

        plt.gca().axis('off')

        plt.savefig(pos_path + f"local_pos_{i}.svg", bbox_inches='tight', pad_inches=0)

        position = states[i, 0:2]
        heading = states[i, 4]
        local_map.plot_local_map_offset(position, heading, map_data.map_origin[:2], map_data.map_resolution, full_path, i)



plot_local_maps("LocalImgs")
# plot_local_maps("Data/devel_local_mpcc/")







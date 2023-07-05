import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd

from LocalMapRacing.DataTools.plotting_utils import *

mpl.rcParams['pdf.use14corefonts'] = True

map_list = ["aut", "esp", "gbr", "mco"]


def load_stats_all_maps(vehicle_name):
    path = f"Data/{vehicle_name}/"
    print(f"Loading data from {path}")
    dfs = []
    for m, map_name in enumerate(map_list):
        file = path + f"Statistics{map_name.upper()}.txt"
        with open(file, 'r') as f:
            line_data = f.readlines()[2].split(',')
        df_entry = {}
        df_entry["planner"] = vehicle_name
        df_entry["map_name"] = map_name
        df_entry["distance"] = float(line_data[1])
        df_entry["progress"] = float(line_data[2]) *100
        df_entry["lap_time"] = float(line_data[3])
        df_entry["avg_speed"] = float(line_data[4])
        df_entry["lap_n"] = int(line_data[0])
        dfs.append(pd.DataFrame(df_entry, index=[m]))

    df = pd.concat(dfs, axis=0)
    print(df)

    return df

def make_lap_time_barplot():
    vehicle_list = ["GlobalPP", "LocalRacePP_1"]
    labels = ["Global", "Local"]

    dfs = [load_stats_all_maps(vehicle_name) for vehicle_name in vehicle_list]
    plt.figure(1, figsize=(4.5, 2))

    xs = np.arange(4)
    barWidth = 0.4
    w = 0.05
    br1 = xs - barWidth/2
    br2 = [x + barWidth for x in br1]
    brs = [br1, br2]

    for v, vehicle_name in enumerate(vehicle_list):
        plt.bar(brs[v], dfs[v]["lap_time"], label=labels[v], color=color_pallette[v], width=barWidth)


    map_labels = [map_name.upper() for map_name in map_list]
    plt.xticks(xs, map_labels)
    plt.ylabel("Lap time (s)")
    plt.legend()

    std_img_saving("Data/Imgs/lap_time_barplot")


make_lap_time_barplot()




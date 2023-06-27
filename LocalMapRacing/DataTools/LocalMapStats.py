import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from LocalMapRacing.DataTools.plotting_utils import *

SEPERATION = 1.2

def analyse_local_map_stats(path):
    map_root = path + "LocalMapData/"
    counter = 0
    map_name = "aut"

    track_lengths = []
    for i in range(1, 260):
        file = map_root + f"local_map_{i}.npy"
        local_track = np.load(file)

        track_lengths.append(len(local_track))

    track_lengths = np.array(track_lengths) * SEPERATION 
    print(f"Mean: {np.mean(track_lengths)}")
    print(f"Std: {np.std(track_lengths)}")
    print(f"Min: {np.min(track_lengths)}")
    print(f"Max: {np.max(track_lengths)}")


    plt.rcParams['pdf.use14corefonts'] = True
    plt.figure(figsize=(3.5, 1.8))
    bins = np.arange(9, 24, 1)
    plt.hist(track_lengths, bins=bins, color=color_pallette[0], density=True, alpha=0.8)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(3))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    plt.xlabel("Track length (m)")
    plt.ylabel("Frequency")
    plt.legend(["AUT"])
    plt.grid(True)


    plt.tight_layout()
    plt.savefig(f"{path}track_length_hist.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{path}track_length_hist.svg", bbox_inches='tight', pad_inches=0)
    # plt.show()

def analyse_local_maps(path):
    map_root = path + "LocalMapData/"
    map_name = "aut"

    track_lengths = []
    for i in range(1, 260):
        file = map_root + f"local_map_{i}.npy"
        local_track = np.load(file)

        plt.pl


    plt.tight_layout()
    plt.savefig(f"{path}track_length_hist.pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{path}track_length_hist.svg", bbox_inches='tight', pad_inches=0)
    # plt.show()



analyse_local_map_stats(f"Data/DataLocalPP/")
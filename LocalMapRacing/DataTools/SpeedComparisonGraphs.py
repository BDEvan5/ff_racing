import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
from LocalMapRacing.DataTools.plotting_utils import *


from LocalMapRacing.DataTools.MapData import MapData
from LocalMapRacing.planner_utils.TrackLine import TrackLine 


class AnalyseTestLapData:
    def __init__(self, vehicle_name):
        self.vehicle_name = vehicle_name
        self.path = None
        self.map_name = None
        self.states = None
        self.actions = None
        self.std_track = None
        
        self.track_progresses = None

    def load_data(self, map_name):
        self.path = f"Data/{self.vehicle_name}/Testing{map_name.upper()}/"
        self.map_name = map_name
        self.std_track = TrackLine(self.map_name, False)

        if not self.load_lap_data(): return
        self.calculate_state_progress()

    def load_lap_data(self):
        try:
            data = np.load(self.path + f"Lap_0_history_{self.vehicle_name}.npy")
        except Exception as e:
            print(e)
            # print(f"No data for: " + f"Lap_{self.lap_n}_history_{self.vehicle_name}_{self.map_name}.npy")
            return 0
        self.states = data[:, :7]
        self.actions = data[:, 7:]
        
        return 1 # to say success
    
    def calculate_state_progress(self):
        progresses = []
        z = 0
        for i in range(len(self.states)):
            p = self.std_track.calculate_progress_percent(self.states[i, 0:2])
            if i < 50 and p > 0.5:
                p = 0 
            progresses.append(p)
            
        self.track_progresses = np.array(progresses) * 100


def make_speed_plots():
    d_global = AnalyseTestLapData("GlobalPP")
    d_local = AnalyseTestLapData("LocalRacePP_1")

    map_list = ["aut", "esp", "gbr", "mco"]
    for map_name in map_list:
        racing_track = TrackLine(map_name, True)
        d_local.load_data(map_name)
        d_global.load_data(map_name)

        plt.figure(figsize=(5, 2), num=1)
        plt.clf()
        ss = racing_track.ss / racing_track.ss[-1] * 100
        plt.plot(ss, racing_track.vs, label="Racing Line", color=color_pallette[3])
        plt.plot(d_global.track_progresses, d_global.states[:, 3], label="Global", color=color_pallette[0])
        plt.plot(d_local.track_progresses, d_local.states[:, 3], label="Local", color=color_pallette[1])

        plt.xlabel("Track Progress (%)")
        plt.ylabel("Speed (m/s)")
        plt.legend(ncol=3, loc="lower center")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"Data/Imgs/SpeedComparison_{map_name}.pdf")
        plt.savefig(f"Data/Imgs/SpeedComparison_{map_name}.svg")


def make_steering_plots():
    d_global = AnalyseTestLapData("GlobalPP")
    d_local = AnalyseTestLapData("LocalRacePP_1")

    map_list = ["aut", "esp", "gbr", "mco"]
    for map_name in map_list:
        racing_track = TrackLine(map_name, True)
        d_local.load_data(map_name)
        d_global.load_data(map_name)

        plt.figure(figsize=(5, 2.2), num=1)
        plt.clf()
        ss = racing_track.ss / racing_track.ss[-1] * 100
        plt.plot(d_global.track_progresses, d_global.states[:, 2], label="Global", color=color_pallette[0])
        plt.plot(d_local.track_progresses, d_local.states[:, 2], label="Local", color=color_pallette[1])

        plt.xlabel("Track Progress (%)")
        plt.ylabel("Steering angle (rad)")
        plt.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"Data/Imgs/SteeringComparison_{map_name}.pdf", bbox_inches='tight', pad_inches=0.0)
        plt.savefig(f"Data/Imgs/SteeringComparison_{map_name}.svg", bbox_inches='tight', pad_inches=0.0)



# make_speed_plots()
make_steering_plots()
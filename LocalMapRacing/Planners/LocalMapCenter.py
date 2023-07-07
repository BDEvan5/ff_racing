import numpy as np 
import matplotlib.pyplot as plt
from LocalMapRacing.planner_utils.VehicleStateHistory import VehicleStateHistory
from LocalMapRacing.planner_utils.TrackLine import TrackLine
from numba import njit  

from LocalMapRacing.local_mapping.local_map_utils import *
from LocalMapRacing.local_mapping.LocalMap import LocalMap
from LocalMapRacing.local_mapping.LocalMapGenerator import LocalMapGenerator
from LocalMapRacing.local_mapping.LocalRaceline import LocalRaceline

from LocalMapRacing.DataTools.MapData import MapData
from matplotlib.collections import LineCollection
np.set_printoptions(precision=4)


LOOKAHEAD_DISTANCE = 1.4
WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 8

VERBOSE = False
VERBOSE = True

class LocalMapCenter:
    def __init__(self, test_name, map_name):
        self.name = test_name
        self.path = f"Data/{test_name}/"
        
        ensure_path_exists(self.path)
        # if VERBOSE:
        self.scan_data_path = self.path + f"ScanData_{map_name.upper()}/"
        ensure_path_exists(self.scan_data_path)
        self.online_lm_path = self.path + f"OnlineMaps_{map_name.upper()}/"
        ensure_path_exists(self.online_lm_path)
        self.lm_smooth_path = self.path + f"SmoothMaps_{map_name.upper()}/"
        ensure_path_exists(self.lm_smooth_path)

        self.vehicle_state_history = VehicleStateHistory(test_name, map_name)
        self.counter = 0
                
        self.local_map_generator = LocalMapGenerator(self.path, map_name)
        self.local_map = None # LocalMap(self.path)

        self.track_line = TrackLine(map_name, False, False)
        self.map_data = MapData(map_name)

        angles = np.linspace(-4.7/2, 4.7/2, 1080)
        self.coses = np.cos(angles)
        self.sines = np.sin(angles)
        
    def plan(self, obs):
        if self.counter % 50 == 0:
            print(f"Counter: {self.counter}")

        self.local_map = self.local_map_generator.generate_line_local_map(np.copy(obs['scans'][0]))
        # self.local_map.apply_required_smoothing(self.counter, self.lm_smooth_path)
        # self.local_map.build_smooth_track(self.counter, self.lm_smooth_path)
        crossing = self.local_map.adjust_center_line_smoothing(self.counter, self.lm_smooth_path)
        if crossing:
            print(f"Crossing detected at {self.counter}")

            # c2 = self.local_map.adjust_center_line_smoothing(self.counter, self.lm_smooth_path)

            # print(f"Fixing result.... --> {c2}")
            # if c2: plt.show()

        position = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        heading = obs['full_states'][0][4]
        
        action, lhd_pt = self.pure_pursuit_center_line()

        self.vehicle_state_history.add_memory_entry(obs, action)

        # if VERBOSE:
        if  self.counter < 150:
        # if self.counter > 50 and self.counter < 150:
            np.save(self.scan_data_path + f"scan_{self.counter}.npy", obs['scans'][0])

            plt.figure(3)
            plt.clf()
            self.map_data.plot_map_img()
            self.map_data.plot_centre_line()
            x, y = self.map_data.xy2rc(obs['poses_x'][0], obs['poses_y'][0])
            s = 12
            plt.arrow(x, y, s*np.cos(heading), s*np.sin(heading), color='red', width=1, head_width = 4)

            lhd_pt = calculate_offset_coords(np.array(lhd_pt), position, heading)
            x, y = self.map_data.xy2rc(lhd_pt[0], lhd_pt[1])
            plt.plot(x, y, 'x', color='green')
            
            scan_xs = obs['scans'][0] * self.coses
            scan_ys = obs['scans'][0] * self.sines
            pts = np.stack((scan_xs, scan_ys), axis=1)
            pts = calculate_offset_coords(pts, position, heading)
            scan_xs, scan_ys = self.map_data.pts2rc(pts)
            plt.plot(scan_xs, scan_ys, '.', color='blue')

            history = np.array(self.vehicle_state_history.states)
            hxs, hys = self.map_data.pts2rc(history[:, :2])
            plt.plot(hxs, hys, '--', color='green')

            # plt.title(f"Local map ({self.counter}): head: {heading:.2f}, pos: {position[0]:.2f}, {position[1]:.2f}")
            plt.title(f"{self.counter}: Action -> {action}")

            self.local_map.plot_local_map_offset(position, heading, self.map_data.map_origin[:2], self.map_data.map_resolution, save_path=self.online_lm_path, counter=self.counter)

            plt.close()


        # plt.pause(0.001)
        # plt.show()

        # if VERBOSE:
        # print(f"{self.counter} --> Action: {action}")

        self.counter += 1
        return action
        
    def pure_pursuit_center_line(self):
        current_progress = np.linalg.norm(self.local_map.track[0, 0:2])
        lookahead = LOOKAHEAD_DISTANCE + current_progress

        lookahead = min(lookahead, self.local_map.s_track[-1]) 
        lookahead_point = interp_2d_points(lookahead, self.local_map.s_track, self.local_map.track[:, 0:2])
        # if VERBOSE:
        #     print(f"Lookahead: {lookahead:.2f}, Lookahead point: ({lookahead_point[0]:.2f}, {lookahead_point[1]:.2f}): Current progress: {current_progress:.2f}")

        steering_angle = get_local_steering_actuation(lookahead_point, LOOKAHEAD_DISTANCE, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        speed = 3
        
        return np.array([steering_angle, speed]), lookahead_point


    def done_callback(self, final_obs):
        self.vehicle_state_history.save_history()
        
        progress = self.track_line.calculate_progress_percent([final_obs['poses_x'][0], final_obs['poses_y'][0]]) * 100
        
        print(f"Lap complete ({self.track_line.map_name.upper()}) --> Time: {final_obs['lap_times'][0]:.2f}, Progress: {progress:.1f}%")
        
     
    
@njit(cache=True)
def calculate_offset_coords(pts, position, heading):
    rotation = np.array([[np.cos(heading), -np.sin(heading)],
                        [np.sin(heading), np.cos(heading)]])
        
    new_pts = (rotation @ pts.T).T + position
    # new_pts = np.matmul(rotation, pts.T).T + position

    return new_pts

@njit(cache=True)
def calculate_speed(delta, f_s=0.8, max_v=7):
    b = 0.523
    g = 9.81
    l_d = 0.329

    if abs(delta) < 0.03:
        return max_v
    if abs(delta) > 0.4:
        return 3

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    V = min(V, max_v)

    return V
    

     
# @njit(cache=True)
# def calculate_speed_limit(delta, friction_limit=1.2):
#     if abs(delta) < 0.03:
#         return MAX_SPEED

#     V = np.sqrt(friction_limit*GRAVITY*WHEELBASE/np.tan(abs(delta)))
#     V = min(V, MAX_SPEED)

#     return V

# @njit(fastmath=False, cache=True)
def get_steering_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle

@njit(fastmath=False, cache=True)
def get_local_steering_actuation(lookahead_point, lookahead_distance, wheelbase):
    waypoint_y = lookahead_point[1]
    if np.abs(waypoint_y) < 1e-6:
        return 0.0
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle
   

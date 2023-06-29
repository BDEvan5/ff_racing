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

np.set_printoptions(precision=4)


LOOKAHEAD_DISTANCE = 1.1
WHEELBASE = 0.33
MAX_STEER = 0.4
MAX_SPEED = 8


class LocalMapPP:
    def __init__(self, test_name, map_name):
        self.name = test_name
        self.path = f"Data/{test_name}/"
        
        ensure_path_exists(self.path)
        ensure_path_exists(self.path + "ScanData/")

        self.vehicle_state_history = VehicleStateHistory(test_name, map_name)
        self.counter = 0
                
        self.local_map_generator = LocalMapGenerator(self.path)
        self.local_map = None # LocalMap(self.path)
        self.local_raceline = LocalRaceline(self.path)

        self.track_line = TrackLine(map_name, False, False)
        self.map_data = MapData(map_name)
        
    def plan(self, obs):
        self.local_map = self.local_map_generator.generate_line_local_map(obs['scans'][0])
        np.save(self.path + "ScanData/" + f"scan_{self.counter}.npy", obs['scans'][0])
        # self.local_raceline.generate_raceline(self.local_map)
        

        plt.figure(3)
        plt.clf()
        self.map_data.plot_map_img()
        x, y = self.map_data.xy2rc(obs['poses_x'][0], obs['poses_y'][0])
        plt.plot(x, y, 'x', color='red')
        
        position = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        # heading = obs['poses_theta'][0]
        heading = obs['full_states'][0][4]

        plt.title(f"Local map ({self.counter}): head: {heading:.2f}, pos: {position[0]:.2f}, {position[1]:.2f}")
        self.local_map.plot_local_map_offset(position, heading, self.map_data.map_origin[:2], self.map_data.map_resolution, save_path=self.path + f"OnlineMaps/", counter=self.counter)

        action = self.pure_pursuit_center_line()
        # action, lhd = self.pure_pursuit_racing_line(obs)

        # plt.pause(0.001)
        # plt.show()
        self.vehicle_state_history.add_memory_entry(obs, action)

        self.counter += 1
        return action
        
    def pure_pursuit_center_line(self):
        current_progress = np.linalg.norm(self.local_map.track[0, 0:2])
        lookahead = LOOKAHEAD_DISTANCE + current_progress

        lookahead = min(lookahead, self.local_map.s_track[-1]) 
        lookahead_point = interp_2d_points(lookahead, self.local_map.s_track, self.local_map.track[:, 0:2])

        steering_angle = get_local_steering_actuation(lookahead_point, LOOKAHEAD_DISTANCE, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        speed = 3
        
        return np.array([steering_angle, speed])

    def pure_pursuit_racing_line(self, obs):
        current_progress = np.linalg.norm(self.local_raceline.raceline[0, :])
        lookahead = LOOKAHEAD_DISTANCE + current_progress
        # lookahead = 0.3 + obs['linear_vels_x'][0] * 0.15 + current_progress
        lookahead = min(lookahead, self.local_raceline.s_track[-1]) 
        lookahead_point = interp_2d_points(lookahead, self.local_raceline.s_track, self.local_raceline.raceline)

        exact_lookahead = np.linalg.norm(lookahead_point)
        steering_angle = get_local_steering_actuation(lookahead_point, exact_lookahead, WHEELBASE)
        steering_angle = np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        speed = np.interp(current_progress, self.local_raceline.s_track, self.local_raceline.vs) #* 0.8
        max_turning_speed = calculate_speed(steering_angle, f_s=0.99, max_v=8)
        # print(f"Speed: {speed:.3f}, Max turning speed: {max_turning_speed:.3f} --> Difference: {(speed - max_turning_speed):.3f}")
        speed = min(speed, max_turning_speed)


        return np.array([steering_angle, speed]), lookahead_point
    
        
    def done_callback(self, final_obs):
        self.vehicle_state_history.save_history()
        
        progress = self.track_line.calculate_progress_percent([final_obs['poses_x'][0], final_obs['poses_y'][0]]) * 100
        
        print(f"Lap complete ({self.track_line.map_name.upper()}) --> Time: {final_obs['lap_times'][0]:.2f}, Progress: {progress:.1f}%")
        
     
@njit(cache=True)
def calculate_speed(delta, f_s=0.8, max_v=7):
    b = 0.523
    g = 9.81
    l_d = 0.329

    if abs(delta) < 0.03:
        return max_v
    if abs(delta) > 0.4:
        return 0

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
   

import numpy as np
import os
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
import glob
from matplotlib.collections import LineCollection
from ff_racing.devel.spline_approximation_unclosed import spline_approximation_unclosed
from scipy.interpolate import splrep, BSpline
from scipy import interpolate
    
def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    
    return xs, ys

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)


KAPPA_BOUND = 0.4
VEHICLE_WIDTH = 0.4
NUMBER_LOCAL_MAP_POINTS = 20
POINT_SEP_DISTANCE = 1.2
ax_max_machine = np.array([[0, 8.5],[8, 8.5]])
ggv = np.array([[0, 8.5, 8.5], [8, 8.5, 8.5]])
MU = 0.54 
V_MAX = 8
VEHICLE_MASS = 3.4

def interpolate_track(points, n_points, s=10):
    el = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cs = np.insert(np.cumsum(el), 0, 0)
    ss = np.linspace(0, cs[-1], n_points)
    tck_x = splrep(cs, points[:, 0], s=s)
    tck_y = splrep(cs, points[:, 1], s=s)
    xs = BSpline(*tck_x)(ss) # get unispaced points
    ys = BSpline(*tck_y)(ss)
    new_points = np.hstack((xs[:, None], ys[:, None]))

    return new_points

class LocalMap:
    def __init__(self, path) -> None:
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)
        self.sines = np.sin(self.angles)
        self.xs, self.ys = None, None 

        self.track = None
        self.el_lengths = None
        self.psi = None
        self.kappa = None
        self.nvecs = None

        self.raceline = None
        self.psi_r = None
        self.kappa_r = None
        self.vs = None
        self.el_lengths_r = None
        self.s_raceline = None

        self.local_map_img_path = path + "LocalMapImgs/"
        self.local_map_img_path_debug = path + "LocalMapImgsD/"
        self.local_map_data_path = path + "LocalMapData/"
        self.raceline_img_path = path + "RacingLineImgs/"
        self.raceline_data_path = path + "RacingLineData/"

        ensure_path_exists(self.local_map_img_path)
        ensure_path_exists(self.local_map_img_path_debug)
        # ensure_path_exists(self.local_map_data_path)
        ensure_path_exists(self.raceline_img_path)
        # ensure_path_exists(self.racing_line_data_path)
        self.counter = 0

    def generate_local_map(self, scan):
        self.counter += 1
        xs = self.coses[scan < 10] * scan[scan < 10]
        ys = self.sines[scan < 10] * scan[scan < 10]
        self.xs = xs[180:-180]
        self.ys = ys[180:-180]

        pts = np.hstack((self.xs[:, None], self.ys[:, None]))
        pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        mid_idx = np.argmax(pt_distances)

        l1_cs = np.cumsum(pt_distances[:mid_idx+1])
        l2_cs = np.cumsum(pt_distances[mid_idx:])

        l1_ss = np.linspace(0, l1_cs[-1], NUMBER_LOCAL_MAP_POINTS)
        l2_ss = np.linspace(0, l2_cs[-1], NUMBER_LOCAL_MAP_POINTS)

        l1_xs, l1_ys = interp_2d_points(l1_ss, l1_cs, pts[:mid_idx+1])
        l2_xs, l2_ys = interp_2d_points(l2_ss, l2_cs, pts[mid_idx+1:])

        c_xs = (l1_xs + l2_xs[::-1])/2
        c_ys = (l1_ys + l2_ys[::-1])/2
        center_line = np.hstack((c_xs[:, None], c_ys[:, None]))
        
        cl_dists = np.linalg.norm(center_line[1:] - center_line[:-1], axis=1)
        cl_cs = np.cumsum(cl_dists)
        cl_cs = np.insert(cl_cs, 0, 0)
        cl_ss = np.linspace(0, cl_cs[-1], NUMBER_LOCAL_MAP_POINTS)
        cl_xs, cl_ys = interp_2d_points(cl_ss, cl_cs, center_line)
        
        center_line = np.hstack((cl_xs[:, None], cl_ys[:, None]))
        ws = np.ones_like(center_line)
        self.track = np.concatenate((center_line, ws), axis=1)
        
        self.calculate_track_heading_and_nvecs()

    def generate_local_map_debug(self, scan):
        self.counter += 1
        plt.figure(5)
        plt.clf()
        fig, axs = plt.subplots(4, 2, num=5)
        xs = self.coses[scan < 10] * scan[scan < 10]
        ys = self.sines[scan < 10] * scan[scan < 10]
        self.xs = xs[180:-180]
        self.ys = ys[180:-180]

        pts = np.hstack((self.xs[:, None], self.ys[:, None]))
        pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        distance_threshold = 1.8 # distance in m for an exception
        inds = np.where(pt_distances > distance_threshold)
        if len(inds[0]) == 0:
            print("Problem: no inds greater than 1.5 m. Check LiDAR scan.....")
        arr_inds = np.arange(len(pt_distances))[inds]
        min_ind = np.min(arr_inds) + 1
        max_ind = np.max(arr_inds) + 1

        mid_idx = np.argmax(pt_distances)

        # l1_cs = np.cumsum(pt_distances[:mid_idx])
        # l2_cs = np.cumsum(pt_distances[mid_idx+1:])
        l1_cs = np.cumsum(pt_distances[:min_ind-1])
        l1_cs = np.insert(l1_cs, 0, 0)
        l2_cs = np.cumsum(pt_distances[max_ind:])
        l2_cs = np.insert(l2_cs, 0, 0)
        
        l1_ss = np.linspace(0, l1_cs[-1], NUMBER_LOCAL_MAP_POINTS)
        l2_ss = np.linspace(0, l2_cs[-1], NUMBER_LOCAL_MAP_POINTS)

        l1_xs, l1_ys = interp_2d_points(l1_ss, l1_cs, pts[:min_ind])
        l2_xs, l2_ys = interp_2d_points(l2_ss, l2_cs, pts[max_ind:])
        
        axs[0, 0].plot(l1_cs, 'o', color='blue')
        axs[0, 0].plot(l1_ss, 'o', color='red')
        axs[1, 0].plot(l1_cs, pts[:min_ind, 0], 'o', color='blue')
        axs[1, 0].plot(l1_ss, l1_xs, 'x', color='red')
        axs[2, 0].plot(l1_cs, pts[:min_ind, 1], 'o', color='blue')
        axs[2, 0].plot(l1_ss, l1_ys, 'x', color='red')
        axs[3, 0].plot(pts[:min_ind, 0], pts[:min_ind, 1], '.', color='blue')
        axs[3, 0].plot(l1_xs, l1_ys, 'x', color='red')   

        axs[0, 1].plot(l2_cs, 'o', color='blue')
        axs[0, 1].plot(l2_ss, 'o', color='red')
        axs[1, 1].plot(l2_cs, pts[max_ind:, 0], 'o', color='blue')
        axs[1, 1].plot(l2_ss, l2_xs, 'x', color='red')
        axs[2, 1].plot(l2_cs, pts[max_ind:, 1], 'o', color='blue')
        axs[2, 1].plot(l2_ss, l2_ys, 'x', color='red')
        axs[3, 1].plot(pts[max_ind:, 0], pts[max_ind:, 1], '.', color='blue')
        axs[3, 1].plot(l2_xs, l2_ys, 'x', color='red')

        plt.pause(0.0001)

        if l1_cs[-1] > l2_cs[-1]:
            long_xs = l1_xs
            long_ys = l1_ys
        else:
            long_xs = l2_xs
            long_ys = l2_ys

        c_xs = (l1_xs + l2_xs[::-1])/2
        c_ys = (l1_ys + l2_ys[::-1])/2
        center_line = np.hstack((c_xs[:, None], c_ys[:, None]))
        
        cl_dists = np.linalg.norm(center_line[1:] - center_line[:-1], axis=1)
        cl_cs = np.cumsum(cl_dists)
        cl_cs = np.insert(cl_cs, 0, 0)
        cl_ss = np.linspace(0, cl_cs[-1], NUMBER_LOCAL_MAP_POINTS)
        cl_xs, cl_ys = interp_2d_points(cl_ss, cl_cs, center_line)
        
        center_line = np.hstack((cl_xs[:, None], cl_ys[:, None]))
        ws = np.ones_like(center_line)
        self.track = np.concatenate((center_line, ws), axis=1)
        
        t2 = np.hstack((long_xs[:, None], long_ys[:, None]))
        el2 = np.linalg.norm(np.diff(t2[:, :2], axis=0), axis=1)
        # print(el2)
        print(t2[-4:])
        psi2, kappa2 = tph.calc_head_curv_num.calc_head_curv_num(t2, el2, False)
        n2 = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi2-np.pi/2)

        c2 = t2 + n2 * 1

        self.calculate_track_heading_and_nvecs()

        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]

        plt.figure(1)
        plt.clf()
        plt.plot(self.xs, self.ys, 'x', color='blue', alpha=0.7)

        plt.plot(l1_xs, l1_ys, '-o', color='green', linewidth=2, markersize=10, alpha=0.6)
        plt.plot(l2_xs, l2_ys, '-o', color='green', linewidth=2, markersize=10, alpha=0.6)
        # plt.plot(long_xs, long_ys, '-o', color='grey', linewidth=1)

        plt.plot(self.track[:, 0], self.track[:, 1], '-', color='red', label="Center", linewidth=3)
        plt.plot(c2[:, 0], c2[:, 1], '-', color='orange', label="Center", linewidth=3)

        for i in range(len(self.track)):
            xs = [c2[i, 0], t2[i, 0]]
            ys = [c2[i, 1], t2[i, 1]]
            plt.plot(xs, ys, 'yellow')

        plt.plot(0, 0, 'x', color='black', label="Origin")

        plt.plot(l1[:, 0], l1[:, 1], color='purple')
        plt.plot(l2[:, 0], l2[:, 1], color='purple')

        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(self.local_map_img_path_debug + f"Local_map_debug_{self.counter}.svg")

        plt.pause(0.0001)
        print("Done")

    def generate_line_local_map(self, scan):
        self.counter += 1
        xs = self.coses[scan < 10] * scan[scan < 10]
        ys = self.sines[scan < 10] * scan[scan < 10]
        self.xs = xs[180:-180]
        self.ys = ys[180:-180]

        pts = np.hstack((self.xs[:, None], self.ys[:, None]))
        pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        track_width = np.linalg.norm(pts[0] - pts[-1]) * 0.95
        distance_threshold = 1.8 # distance in m for an exception
        inds = np.where(pt_distances > distance_threshold)
        if len(inds[0]) == 0:
            print("Problem: no inds greater than 1.5 m. Check LiDAR scan.....")
        arr_inds = np.arange(len(pt_distances))[inds]
        min_ind = np.min(arr_inds) + 1
        max_ind = np.max(arr_inds) + 1

        l1_cs = np.cumsum(pt_distances[:min_ind-1])
        l2_cs = np.cumsum(pt_distances[max_ind:])

        if l1_cs[-1] > l2_cs[-1]:
            long_pts = pts[:min_ind]
            line_length = l1_cs[-1]
            w = 1
        else:
            long_pts = pts[max_ind:]
            long_pts = long_pts[::-1]
            line_length = l2_cs[-1]
            w = -1
            
        n_pts = int(line_length / POINT_SEP_DISTANCE)
        long_side = interpolate_track(long_pts, n_pts*5, 2)
        long_side = interpolate_track(long_side, n_pts, 2)

        side_el = np.linalg.norm(np.diff(long_side[:, :2], axis=0), axis=1)
        psi2, kappa2 = tph.calc_head_curv_num.calc_head_curv_num(long_side, side_el, False)
        side_nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi2-np.pi/2)

        center_line = long_side + side_nvecs * w * track_width / 2
        center_line = interpolate_track(center_line, n_pts, 2)

        ws = np.ones_like(center_line) * track_width / 2
        self.track = np.concatenate((center_line, ws), axis=1)
        self.calculate_track_heading_and_nvecs()

        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, 4)
        while crossing:
            self.track[:, 2:] *= 0.9
            self.calculate_track_heading_and_nvecs()
            crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs, 4)
            print("Normals crossed --> New Crossing: ", crossing)

        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]

        plt.figure(1)
        plt.clf()
        plt.plot(self.xs, self.ys, 'x', color='blue', alpha=0.7)

        plt.plot(long_side[:, 0], long_side[:, 1], '-o', color='green', linewidth=1, markersize=10, alpha=0.8)

        plt.plot(self.track[:, 0], self.track[:, 1], '-o', color='red', label="Center", linewidth=3)

        plt.plot(0, 0, 'x', color='black', label="Origin", markersize=10)

        plt.plot(l1[:, 0], l1[:, 1], color='purple')
        plt.plot(l2[:, 0], l2[:, 1], color='purple')

        for i in range(len(self.track)):
            xs = [l1[i, 0], l2[i, 0]]
            ys = [l1[i, 1], l2[i, 1]]
            plt.plot(xs, ys, 'yellow')

        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(self.local_map_img_path_debug + f"Local_map_debug_{self.counter}.svg")


        # plt.pause(0.0001)


    def calculate_track_heading_and_nvecs(self):
        self.el_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track[:, :2], self.el_lengths, False)
        
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi-np.pi/2)
        
    def generate_minimum_curvature_path(self):
        coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(self.track[:, :2], self.el_lengths, self.psi[0], self.psi[-1])
        self.psi = self.psi - np.pi/2 #! check this, it does not look right

        # Todo: adjust the start point to be the vehicle location and adjust the width accordingly
        try:
            alpha, error = tph.opt_min_curv.opt_min_curv(self.track, self.nvecs, M, KAPPA_BOUND, VEHICLE_WIDTH, print_debug=False, closed=False, psi_s=self.psi[0], psi_e=self.psi[-1], fix_s=True)

            raceline = self.track[:, :2] + np.expand_dims(alpha, 1) * self.nvecs
        except:
            print("Error in optimising min curvature path")
            raceline = self.track[:, :2]
        
        self.raceline, self.s_raceline = normalise_raceline(raceline, 0.2, self.psi)
        self.el_lengths_r = np.diff(self.s_raceline)

        self.psi_r, self.kappa_r = tph.calc_head_curv_num.calc_head_curv_num(self.raceline, self.el_lengths_r, False)

    def generate_max_speed_profile(self, starting_speed=V_MAX):
        mu = MU * np.ones_like(self.kappa_r) 
        
        self.vs = tph.calc_vel_profile.calc_vel_profile(ax_max_machine, self.kappa_r, self.el_lengths_r, False, 0, VEHICLE_MASS, ggv=ggv, mu=mu, v_max=V_MAX, v_start=starting_speed)

    def plot_save_local_map(self):
        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]

        plt.figure(1)
        plt.clf()
        plt.plot(self.xs, self.ys, '.', color='blue', alpha=0.7)
        plt.plot(self.track[:, 0], self.track[:, 1], '-', color='red', label="Center", linewidth=3)
        plt.plot(0, 0, 'x', color='black', label="Origin")

        plt.plot(l1[:, 0], l1[:, 1], color='green')
        plt.plot(l2[:, 0], l2[:, 1], color='green')

        for i in range(len(self.track)):
            xs = [l1[i, 0], l2[i, 0]]
            ys = [l1[i, 1], l2[i, 1]]
            plt.plot(xs, ys, 'orange')

        plt.title("Local Map")

        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(self.local_map_img_path + f"Local_map_{self.counter}.svg")


        
    def plot_save_raceline(self):
        plt.figure(1)
        plt.clf()
        plt.title("Racing Line Velocity Profile")

        plt.plot(self.track[:, 0], self.track[:, 1], '-', linewidth=2, color='blue')

        vs = self.vs
        points = self.raceline.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(2, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(3)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line)
        plt.gca().set_aspect('equal', adjustable='box')

        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]
        plt.plot(l1[:, 0], l1[:, 1], color='green')
        plt.plot(l2[:, 0], l2[:, 1], color='green')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.legend(["Track", "Raceline", "Boundaries"], ncol=3)

        plt.savefig(self.raceline_img_path + f"Raceline_{self.counter}.svg")




def normalise_raceline(raceline, step_size, psis):
    r_el_lengths = np.linalg.norm(np.diff(raceline, axis=0), axis=1)
    
    coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(raceline, r_el_lengths, psis[0], psis[-1])
    
    spline_lengths_raceline = tph.calc_spline_lengths.            calc_spline_lengths(coeffs_x=coeffs_x, coeffs_y=coeffs_y)
    
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = tph.            interp_splines.interp_splines(spline_lengths=spline_lengths_raceline,
                                    coeffs_x=coeffs_x,
                                    coeffs_y=coeffs_y,
                                    incl_last_point=False,
                                    stepsize_approx=0.2)
    
    return raceline_interp, s_raceline_interp



def run_loop(path="Data/LocalMapPlanner2/"):
    laps = glob.glob(path + "ScanData/LocalMapPlanner2_*.npy")
    laps.sort()

    local_map = LocalMap(path)
    for i, lap in enumerate(laps):
        print(f"Processing lap {i}")
        data = np.load(lap)

        # local_map.generate_local_map_debug(data)
        local_map.generate_line_local_map(data)
        # local_map.plot_save_local_map()
        
        # if i > 50:
        #     break
# 


if __name__ == "__main__":
    run_loop()
    plt.show()
    pass
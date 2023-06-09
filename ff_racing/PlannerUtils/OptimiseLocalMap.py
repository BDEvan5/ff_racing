import numpy as np
import matplotlib.pyplot as plt
import ff_racing.tph_utils as tph
from matplotlib.collections import LineCollection
np.set_printoptions(precision=4)
from ff_racing.PlannerUtils.local_map_utils import *


DISTNACE_THRESHOLD = 1.6 # distance in m for an exception
TRACK_WIDTH = 1.8 # use fixed width
POINT_SEP_DISTANCE = 1.2
KAPPA_BOUND = 0.4
VEHICLE_WIDTH = 0.8
ax_max_machine = np.array([[0, 8.5],[8, 8.5]])
ggv = np.array([[0, 8.5, 8.5], [8, 8.5, 8.5]])
MU = 0.5
V_MAX = 8
VEHICLE_MASS = 3.4

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
        self.s_track = None

        self.raceline = None
        self.psi_r = None
        self.kappa_r = None
        self.vs = None
        self.el_lengths_r = None
        self.s_raceline = None

        self.local_map_img_path = path + "LocalMapImgs/"
        self.local_map_data_path = path + "LocalMapData/"
        self.raceline_img_path = path + "RacingLineImgs/"
        self.raceline_data_path = path + "RacingLineData/"

        ensure_path_exists(self.local_map_img_path)
        # ensure_path_exists(self.local_map_data_path)
        ensure_path_exists(self.raceline_img_path)
        # ensure_path_exists(self.racing_line_data_path)
        self.counter = 0

    def generate_line_local_map(self, scan):
        self.counter += 1
        xs = self.coses[scan < 10] * scan[scan < 10] #? why are long beams excluded???? Try without this.
        ys = self.sines[scan < 10] * scan[scan < 10]

        pts, pt_distances, inds = self.extract_track_lines(xs, ys)

        long_side, n_pts, w = self.calculate_longest_line(pts, pt_distances, inds)

        track = self.project_side_to_track(long_side, w, n_pts)
        track, el, ss, psi, kappa, nvecs = self.adjust_track_normals(track)

        self.track = track
        self.el_lengths = el
        self.s_track = ss
        self.psi = psi
        self.kappa = kappa
        self.nvecs = nvecs

    def extract_track_lines(self, xs, ys):
        clip_xs, clip_ys = xs[180:-180], ys[180:-180] 

        pts = np.hstack((clip_xs[:, None], clip_ys[:, None]))
        pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        inds = np.where(pt_distances > DISTNACE_THRESHOLD)
        inds = np.delete(inds, np.where(inds[0] == 0)) 

        if len(inds) == 0:
            pts = np.hstack((xs[:, None], ys[:, None]))
            pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            inds = np.where(pt_distances > DISTNACE_THRESHOLD)
            inds = np.delete(inds, np.where(inds[0] == 0)) 

            if len(inds) == 0: 
                raise IOError("Problem with full scan, no gaps found")

        return pts, pt_distances, inds

    def calculate_longest_line(self, pts, pt_distances, inds):
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
        long_side = interpolate_track(long_pts, n_pts*2, 0)

        return long_side, n_pts, w

    def calculate_length_heading_nvecs(self, pts):
        el_lengths = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        ss = np.insert(np.cumsum(el_lengths), 0, 0)
        psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(pts, el_lengths, False)
        nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(psi-np.pi/2)

        return el_lengths, ss, psi, kappa, nvecs

    def project_side_to_track(self, side, w, n_pts):
        _el, _ss, _psi, _kappa, side_nvecs = self.calculate_length_heading_nvecs(side)
        center_line = side + side_nvecs * w * TRACK_WIDTH / 2
        center_line = interpolate_track(center_line, n_pts, 1)

        ws = np.ones_like(center_line) * TRACK_WIDTH / 2
        track = np.concatenate((center_line, ws), axis=1)

        return track

    def adjust_track_normals(self, track):
        el, ss, psi, kappa, nvecs = self.calculate_length_heading_nvecs(track)

        crossing_horizon = min(5, len(track)//2 -1)
        i = 0
        while i < 20 and tph.check_normals_crossing.check_normals_crossing(track, nvecs, crossing_horizon):
            i += 1
            if np.mean(self.kappa) > 0:
                track[:, 2] *= 0.9
            else:
                track[:, 3] *= 0.9
            el, ss, psi, kappa, nvecs = self.calculate_length_heading_nvecs(track)
            print(f"{i}:: Normals crossed --> New width: {track[0, 2:]}")

        return track, el, ss, psi, kappa, nvecs

    def calculate_track_heading_and_nvecs(self):
        self.el_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track[:, :2], self.el_lengths, False)
        
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi-np.pi/2)

        self.s_track = np.cumsum(self.el_lengths)
        self.s_track = np.insert(self.s_track, 0, 0)
        
    def generate_minimum_curvature_path(self):
        coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(self.track[:, :2], self.el_lengths, self.psi[0], self.psi[-1])
        self.psi = self.psi - np.pi/2 

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
        plt.plot(self.xs, self.ys, '.', color='#0057e7', alpha=0.7)
        plt.plot(self.track[:, 0], self.track[:, 1], '-', color='#E74C3C', label="Center", linewidth=3)
        plt.plot(0, 0, 'x', color='black', markersize=10)

        plt.plot(l1[:, 0], l1[:, 1], color='#ffa700')
        plt.plot(l2[:, 0], l2[:, 1], color='#ffa700')

        for i in range(len(self.track)):
            xs = [l1[i, 0], l2[i, 0]]
            ys = [l1[i, 1], l2[i, 1]]
            plt.plot(xs, ys, '#ffa700')

        plt.title("Local Map")

        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig(self.local_map_img_path + f"Local_map_{self.counter}.svg")
        
    def plot_save_raceline(self, lookahead_point=None):
        plt.figure(1)
        plt.clf()
        plt.title("Racing Line Velocity Profile")

        plt.plot(self.track[:, 0], self.track[:, 1], '--', linewidth=2, color='black')

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

        plt.plot(0, 0, 'x', markersize=10, color='black')
        if lookahead_point is not None:
            plt.plot(lookahead_point[0], lookahead_point[1], 'ro')

        # plt.xticks([])
        # plt.yticks([])
        plt.tight_layout()
        # plt.legend(["Track", "Raceline", "Boundaries"], ncol=3)

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



if __name__ == "__main__":
    pass
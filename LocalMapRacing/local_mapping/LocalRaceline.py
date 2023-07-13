import numpy as np
import matplotlib.pyplot as plt
import LocalMapRacing.tph_utils as tph
from matplotlib.collections import LineCollection
np.set_printoptions(precision=4)
from LocalMapRacing.local_mapping.local_map_utils import *
from scipy import interpolate, optimize

KAPPA_BOUND = 0.4
VEHICLE_WIDTH = 0.8
ax_max_machine = np.array([[0, 8.5],[8, 8.5]])
ggv = np.array([[0, 8.5, 8.5], [8, 8.5, 8.5]])
MU = 0.5
V_MAX = 8
VEHICLE_MASS = 3.4

# KAPPA_BOUND = 0.3
# VEHICLE_WIDTH = 0.1
# a_max = 38
# ax_max_machine = np.array([[0, a_max],[8, a_max]])
# ggv = np.array([[0, a_max, a_max], [8, a_max, a_max]])
# MU = 0.6

class LocalRaceline:
    def __init__(self, path):
        self.lm = None

        self.raceline = None
        self.el_lengths = None
        self.s_track = None
        self.psi = None
        self.kappa = None
        self.vs = None

        self.raceline_data_path = path + "RacingLineData/"
        # ensure_path_exists(self.racing_line_data_path)

    def generate_raceline(self, local_map):
        self.lm = local_map

        raceline = self.generate_minimum_curvature_path()
        self.normalise_raceline(raceline)
        self.generate_max_speed_profile()

        raceline = np.concatenate([self.raceline, self.vs[:, None]], axis=-1)
        
        self.tck = interpolate.splprep([self.raceline[:, 0], self.raceline[:, 1]], k=3, s=0)[0]
        
        return raceline

    def generate_minimum_curvature_path(self):
        coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(self.lm.track[:, :2], self.lm.el_lengths, self.lm.psi[0], self.lm.psi[-1])
        psi = self.lm.psi - np.pi/2 # Why?????

        try:
            alpha, error = tph.opt_min_curv.opt_min_curv(self.lm.track, self.lm.nvecs, M, KAPPA_BOUND, VEHICLE_WIDTH, print_debug=False, closed=False, psi_s=psi[0], psi_e=psi[-1], fix_s=True)#, fix_e=True)

            raceline = self.lm.track[:, :2] + np.expand_dims(alpha, 1) * self.lm.nvecs
        except Exception as e:
            print("Error in optimising min curvature path")
            print(f"Exception: {e}")
            raceline = self.lm.track[:, :2]

        return raceline

    def normalise_raceline(self, raceline):
        self.raceline, self.s_track = normalise_raceline(raceline, 0.2, self.lm.psi-np.pi/2)
        self.el_lengths_r = np.diff(self.s_track)

        self.psi_r, self.kappa_r = tph.calc_head_curv_num.calc_head_curv_num(self.raceline, self.el_lengths_r, False)

    def generate_max_speed_profile(self, starting_speed=V_MAX):
        mu = MU * np.ones_like(self.kappa_r) 

        self.vs = tph.calc_vel_profile.calc_vel_profile(ax_max_machine, self.kappa_r, self.el_lengths_r, False, 0, VEHICLE_MASS, ggv=ggv, mu=mu, v_max=V_MAX, v_start=starting_speed)


    def calculate_s(self, point):
        dists = np.linalg.norm(point - self.raceline[:, :2], axis=1)
        t_guess = self.s_track[np.argmin(dists)] / self.s_track[-1]

        t_point = optimize.fmin(dist_to_p, x0=t_guess, args=(self.tck, point), disp=False)
        interp_return = interpolate.splev(t_point, self.tck, ext=3)
        closest_pt = np.array(interp_return).T
        if len(closest_pt.shape) > 1: closest_pt = closest_pt[0]

        return closest_pt, t_point

    def calculate_lookahead_point(self, lookahead_distance):
        track_pt, current_s = self.calculate_s([0, 0])
        lookahead_s = current_s + lookahead_distance / self.s_track[-1]

        lookahead_pt = np.array(interpolate.splev(lookahead_s, self.tck, ext=3)).T
        if len(lookahead_pt.shape) > 1: lookahead_pt = lookahead_pt[0]

        speed = np.interp(current_s, self.s_track/self.s_track[-1], self.vs)[0]

        return lookahead_pt, speed


def normalise_raceline(raceline, step_size, psis):
    r_el_lengths = np.linalg.norm(np.diff(raceline, axis=0), axis=1)
    
    coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(raceline, r_el_lengths, psis[0], psis[-1])
    
    spline_lengths_raceline = tph.calc_spline_lengths.            calc_spline_lengths(coeffs_x=coeffs_x, coeffs_y=coeffs_y)
    
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = tph.            interp_splines.interp_splines(spline_lengths=spline_lengths_raceline,
                                    coeffs_x=coeffs_x,
                                    coeffs_y=coeffs_y,
                                    incl_last_point=False,
                                    stepsize_approx=step_size)
    
    return raceline_interp, s_raceline_interp


class PlotLocalRaceline(LocalRaceline):
    def __init__(self, path):
        super().__init__(path)

        self.raceline_img_path = path + "RacingLineImgs/"
        ensure_path_exists(self.raceline_img_path)
    
    def plot_save_raceline(self, lookahead_point=None, counter=0):
        plt.figure(1)
        plt.clf()
        plt.title("Racing Line Velocity Profile")

        plt.plot(self.lm.track[:, 0], self.lm.track[:, 1], '--', linewidth=2, color='black')

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

        l1 = self.lm.track[:, :2] + self.lm.nvecs * self.lm.track[:, 2][:, None]
        l2 = self.lm.track[:, :2] - self.lm.nvecs * self.lm.track[:, 3][:, None]
        plt.plot(l1[:, 0], l1[:, 1], color='green')
        plt.plot(l2[:, 0], l2[:, 1], color='green')

        plt.plot(0, 0, 'x', markersize=10, color='black')
        if lookahead_point is not None:
            plt.plot(lookahead_point[0], lookahead_point[1], 'ro')

        plt.tight_layout()

        plt.savefig(self.raceline_img_path + f"Raceline_{counter}.svg")





import numpy as np
import matplotlib.pyplot as plt
import ff_racing.tph_utils as tph
from matplotlib.collections import LineCollection
np.set_printoptions(precision=4)
from ff_racing.local_mapping.local_map_utils import *


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

class LocalRaceline:
    def __init__(self, path):
        self.lm = None

        self.raceline = None
        self.el_lengths = None
        self.s_track = None
        self.psi = None
        self.kappa = None
        self.vs = None

        self.raceline_img_path = path + "RacingLineImgs/"
        self.raceline_data_path = path + "RacingLineData/"

        ensure_path_exists(self.raceline_img_path)
        # ensure_path_exists(self.racing_line_data_path)

    def generate_raceline(self, local_map):
        self.lm = local_map

        raceline = self.generate_minimum_curvature_path()
        self.normalise_raceline(raceline)
        self.generate_max_speed_profile()

    def generate_minimum_curvature_path(self):
        coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(self.lm.track[:, :2], self.lm.el_lengths, self.lm.psi[0], self.lm.psi[-1])
        psi = self.lm.psi - np.pi/2 # Why?????

        # Todo: adjust the start point to be the vehicle location and adjust the width accordingly
        try:
            alpha, error = tph.opt_min_curv.opt_min_curv(self.lm.track, self.lm.nvecs, M, KAPPA_BOUND, VEHICLE_WIDTH, print_debug=False, closed=False, psi_s=psi[0], psi_e=psi[-1], fix_s=True)

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



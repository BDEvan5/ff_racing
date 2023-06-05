import numpy as np
import os
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
import glob
from matplotlib.collections import LineCollection
    
def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    
    return xs, ys

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)


KAPPA_BOUND = 0.4
VEHICLE_WIDTH = 0.4
NUMBER_LOCAL_MAP_POINTS = 10
ax_max_machine = np.array([[0, 8.5],[8, 8.5]])
ggv = np.array([[0, 8.5, 8.5], [8, 8.5, 8.5]])
MU = 0.54 
V_MAX = 8
VEHICLE_MASS = 3.4

class LocalMap:
    def __init__(self) -> None:
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)
        self.sines = np.sin(self.angles)

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

    def generate_local_map(self, scan):
        xs = self.coses[scan < 10] * scan[scan < 10]
        ys = self.sines[scan < 10] * scan[scan < 10]
        xs = xs[180:-180]
        ys = ys[180:-180]

        pts = np.hstack((xs[:, None], ys[:, None]))
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


        
    def plot_local_raceline(self):
        plt.figure(1)
        plt.clf()
        plt.title("Racing Line Velocity Profile")

        plt.plot(self.pts[:, 0], self.pts[:, 1], '-', linewidth=2, color='blue')

        vs = self.vs
        points = self.raceline.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(3)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line)
        plt.gca().set_aspect('equal', adjustable='box')

        ns = self.nvecs 
        ws = np.ones_like(self.nvecs) * self.ws[:, None]
        l_line = self.pts - np.array([ns[:, 0] * ws[:, 0], ns[:, 1] * ws[:, 0]]).T
        r_line = self.pts + np.array([ns[:, 0] * ws[:, 1], ns[:, 1] * ws[:, 1]]).T

        plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1, color='green')
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1, color='green')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.legend(["Track", "Raceline", "Boundaries"], ncol=3)

        # plt.show()
        plt.pause(0.0001)






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
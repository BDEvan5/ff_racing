import numpy as np
import os
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
    
def interp_2d_points(ss, xp, points):
    xs = np.interp(ss, xp, points[:, 0])
    ys = np.interp(ss, xp, points[:, 1])
    
    return xs, ys

def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)



class LocalMap:
    def __init__(self, pts, ws) -> None:
        self.xs = pts[:, 0]
        self.ys = pts[:, 1]
        self.pts = pts
        self.ws = ws
        
        self.el_lengths = None
        self.psi = None
        self.kappa = None
        self.nvecs = None
        self.calculate_nvecs()
        
        self.distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        self.lengths = np.insert(np.cumsum(self.distances), 0, 0)
        
        self.t_pts = None
        
    def calculate_nvecs(self):
        self.el_lengths = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.pts, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi-np.pi/2)
        
    def get_lookahead_point(self, lookahead_distance):
        lookahead = min(lookahead, self.lengths[-1]) 
         
        lookahead_point = interp_2d_points(lookahead, self.lengths, self.pts)
        
        return lookahead_point
    
    def plot_map(self):
        l1 = self.pts + self.nvecs * self.ws[:, None]
        l2 = self.pts - self.nvecs * self.ws[:, None]
        
        plt.figure(1)
        plt.clf()
        plt.plot(self.pts[:, 0], self.pts[:, 1], '-', color='red', label="Center", linewidth=3)
        plt.plot(0, 0, 'x', color='black', label="Origin")

        plt.plot(l1[:, 0], l1[:, 1], color='green')
        plt.plot(l2[:, 0], l2[:, 1], color='green')

        for i in range(len(self.ws)):
            xs = [l1[i, 0], l2[i, 0]]
            ys = [l1[i, 1], l2[i, 1]]
            plt.plot(xs, ys)

        # plt.xlim(-1, 10)
        # plt.ylim(-10, 10)

        plt.gca().set_aspect('equal', adjustable='box')
        
    # def generate_trajectory(self):
    #     pass
     
    def generate_minimum_curvature_path(self):
        coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(self.pts, self.el_lengths, self.psi[0], self.psi[-1])
        # print(self.psi)
        self.psi = self.psi - np.pi/2

        #! these are parameters....
        kappa_bound = 0.8
        width = 0.01
        A = M
        track = np.concatenate((self.pts, self.ws[:, None], self.ws[:, None]), axis=1)
        #! Todo: adjust the start point to be the vehicle location and adjust the width accordingly
        alpha, error = tph.opt_min_curv.opt_min_curv(track, self.nvecs, A, kappa_bound, width, print_debug=True, closed=False, psi_s=self.psi[0], psi_e=self.psi[-1], fix_s=True)

        # raceline_interp, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp,            t_values_raceline_interp, s_raceline_interp, spline_lengths_raceline, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(self.pts, self.nvecs, alpha, 0.2) # 0.2 is the sampling rate...
        self.raceline = self.pts + np.expand_dims(alpha, 1) * self.nvecs


        # self.ss = s_raceline_interp
        # self.raceline = raceline_interp
        # self.psi_r, self.kappa_r = tph.calc_head_curv_num.calc_head_curv_num(self.raceline, el_lengths_raceline_interp_cl, True)

        plt.figure(3)
        plt.clf()
        plt.title("Minimum Curvature Raceline")
        plt.plot(track[:, 0], track[:, 1], '-', linewidth=2, color='blue')
        plt.plot(self.raceline[:, 0], self.raceline[:, 1], 'x-', linewidth=2, color='red')

        ns = self.nvecs 
        ws = np.ones_like(self.nvecs) * self.ws[:, None]
        l_line = self.pts - np.array([ns[:, 0] * ws[:, 0], ns[:, 1] * ws[:, 0]]).T
        r_line = self.pts + np.array([ns[:, 0] * ws[:, 1], ns[:, 1] * ws[:, 1]]).T

        plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1, color='green')
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1, color='green')
        
        plt.legend(["Track", "Raceline", "Boundaries"])
        plt.gca().set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.pause(0.001)   
        # plt.show()

def create_raceline_segment(refline, normvectors, alpha, step_size=0.2):
    raceline = refline + np.expand_dims(alpha, 1) * normvectors

    coeffs_x_raceline, coeffs_y_raceline, A_raceline, normvectors_raceline = tph.calc_splines.\
        calc_splines(path=raceline_cl,
                     use_dist_scaling=False)

    # calculate new spline lengths
    spline_lengths_raceline = tph.calc_spline_lengths. \
        calc_spline_lengths(coeffs_x=coeffs_x_raceline,
                            coeffs_y=coeffs_y_raceline)

    # interpolate splines for evenly spaced raceline points
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = tph.\
        interp_splines.interp_splines(spline_lengths=spline_lengths_raceline,
                                      coeffs_x=coeffs_x_raceline,
                                      coeffs_y=coeffs_y_raceline,
                                      incl_last_point=False,
                                      stepsize_approx=stepsize_interp)

    # calculate element lengths
    s_tot_raceline = float(np.sum(spline_lengths_raceline))
    el_lengths_raceline_interp = np.diff(s_raceline_interp)
    el_lengths_raceline_interp_cl = np.append(el_lengths_raceline_interp, s_tot_raceline - s_raceline_interp[-1])

    return raceline_interp, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, \
           t_values_raceline_interp, s_raceline_interp, spline_lengths_raceline, el_lengths_raceline_interp_cl


import glob
def run_loop(path="Data/LocalMapPlanner/LocalMapData/"):
    laps = glob.glob(path + "local_map_*.npy")
    laps.sort()
    # print(laps)
    
    for i, lap in enumerate(laps):
        data = np.load(lap)
        local_map = LocalMap(data[:, :2], data[:, 2])
        local_map.generate_minimum_curvature_path()




if __name__ == "__main__":
    run_loop()
    plt.show()
    pass
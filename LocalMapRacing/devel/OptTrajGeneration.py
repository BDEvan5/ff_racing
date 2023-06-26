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



class LocalMap:
    def __init__(self, pts, ws, counter) -> None:
        self.xs = pts[:, 0]
        self.ys = pts[:, 1]
        self.counter = counter
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
        #! add the self inits here
        
    def calculate_nvecs(self):
        self.el_lengths = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.pts, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi-np.pi/2)
        
    def get_lookahead_point(self, lookahead_distance):
        lookahead = min(lookahead_distance, self.lengths[-1]) 
         
        lookahead_point = interp_2d_points(lookahead, self.lengths, self.pts)
        
        return lookahead_point
    
    def generate_minimum_curvature_path(self):
        coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(self.pts, self.el_lengths, self.psi[0], self.psi[-1])
        self.psi = self.psi - np.pi/2

        kappa_bound = 0.4
        width = 0.1
        A = M
        track = np.concatenate((self.pts, self.ws[:, None], self.ws[:, None]), axis=1)
        #! Todo: adjust the start point to be the vehicle location and adjust the width accordingly
        alpha, error = tph.opt_min_curv.opt_min_curv(track, self.nvecs, A, kappa_bound, width, print_debug=False, closed=False, psi_s=self.psi[0], psi_e=self.psi[-1], fix_s=True)

        raceline = self.pts + np.expand_dims(alpha, 1) * self.nvecs
        
        raceline_interp, s_raceline_interp, el_lengths_raceline_interp_cl = normalise_raceline(raceline, 0.2, self.psi)

        self.ss = s_raceline_interp
        self.raceline = raceline_interp
        self.psi_r, self.kappa_r = tph.calc_head_curv_num.calc_head_curv_num(self.raceline, el_lengths_raceline_interp_cl[:-1], False)

    def generate_max_speed_profile(self):
        ax_max_machine = np.array([[0, 8.5],[8, 8.5]])
        ggv = np.array([[0, 8.5, 8.5], [8, 8.5, 8.5]])
        mu = 0.5 * np.ones(len(self.kappa_r)) # this is why my race lines are so slow......
        v_max = 8
        m_vehicle = 3.4
        el_lengths = np.linalg.norm(np.diff(self.raceline, axis=0), axis=1)
        
        
        plt.figure(3)
        plt.clf()
        plt.plot(self.kappa_r)
        plt.plot(np.abs(self.kappa_r))
        plt.title("Curvature Profile")
        plt.ylabel("Curvature (rad/s)")
        plt.ylim([-0.3, 0.3])
        plt.grid(True)
        plt.pause(0.01)

        self.vs = tph.calc_vel_profile.calc_vel_profile(ax_max_machine, self.kappa_r, el_lengths, False, 0, m_vehicle, ggv=ggv, mu=mu, v_max=v_max, v_start=v_max)

        plt.figure(2)
        plt.clf()
        plt.plot(self.vs)
        plt.title("Velocity Profile")
        plt.ylabel("Velocity (m/s)")
        plt.grid(True)
        plt.ylim([0, 8.5])
        plt.pause(0.01)

        ts = tph.calc_t_profile.calc_t_profile(self.vs, el_lengths, 0)
        print(f"Time: {ts[-1]}")

        self.acc = tph.calc_ax_profile.calc_ax_profile(self.vs, el_lengths, True)

    def save_racing_line(self):
        save_arr = np.concatenate([self.ss[:, None], self.raceline, self.psi_r[:, None], self.kappa_r[:, None], self.vs[:, None], self.acc[:, None]], axis=1)

        np.savetxt("maps/"+ self.map_name+ '_raceline.csv', save_arr, delimiter=',')

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

        plt.gca().set_aspect('equal', adjustable='box')
        
    def plot_speed_profile(self):
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
        plt.pause(1)

    def plot_raceline(self):
        plt.figure(3)
        plt.clf()
        plt.title("Minimum Curvature Raceline")
        plt.plot(self.pts[:, 0], self.pts[:, 1], '-', linewidth=2, color='blue')
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
        path = "Data/LocalMapPlanner/OptimalCurves/"
        ensure_path_exists(path)
        plt.savefig(path + f"OptimalCurves_{self.counter}.svg")
        # plt.pause(0.001)   
        # plt.show()


def normalise_raceline(raceline, step_size, psis):
    r_el_lengths = np.linalg.norm(np.diff(raceline, axis=0), axis=1)
    
    coeffs_x, coeffs_y, M, normvec_normalized = tph.calc_splines.calc_splines(raceline, r_el_lengths, psis[0], psis[-1])
    
    spline_lengths_raceline = tph.calc_spline_lengths.            calc_spline_lengths(coeffs_x=coeffs_x, coeffs_y=coeffs_y)
    
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = tph.            interp_splines.interp_splines(spline_lengths=spline_lengths_raceline,
                                    coeffs_x=coeffs_x,
                                    coeffs_y=coeffs_y,
                                    incl_last_point=False,
                                    stepsize_approx=0.2)
    
    s_tot_raceline = float(np.sum(spline_lengths_raceline))
    el_lengths_raceline_interp = np.diff(s_raceline_interp)
    el_lengths_raceline_interp_cl = np.append(el_lengths_raceline_interp, s_tot_raceline - s_raceline_interp[-1])
    
    return raceline_interp, s_tot_raceline, el_lengths_raceline_interp_cl

def run_loop(path="Data/LocalMapPlanner/LocalMapData/"):
    laps = glob.glob(path + "local_map_*.npy")
    laps.sort()
    # print(laps)
    
    for i, lap in enumerate(laps):
        print(f"Processing lap {i}")
        data = np.load(lap)
        local_map = LocalMap(data[:, :2], data[:, 2], i)
        local_map.generate_minimum_curvature_path()
        # local_map.plot_raceline()
        
        local_map.generate_max_speed_profile()
        local_map.plot_speed_profile()

        if i > 20:
            break



if __name__ == "__main__":
    run_loop()
    plt.show()
    pass
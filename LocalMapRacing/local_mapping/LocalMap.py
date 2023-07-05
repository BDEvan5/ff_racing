import numpy as np
import LocalMapRacing.tph_utils as tph
import matplotlib.pyplot as plt


class LocalMap:
    def __init__(self, track):
        self.track = track
        self.el_lengths = None
        self.psi = None
        self.kappa = None
        self.nvecs = None
        self.s_track = None

        self.calculate_length_heading_nvecs()

    def calculate_length_heading_nvecs(self):
        self.el_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        self.s_track = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi-np.pi/2)

    def calculate_s(self, point):
        distances = np.linalg.norm(self.track[:, 0:2] - point, axis=1)
        idx = np.argmin(distances)
        x, h = self.interp_pts(idx, distances)
        s = (self.s_track[idx] + x) 

        return s, h

    def interp_pts(self, idx, dists):
        d_ss = self.s_track[idx+1] - self.s_track[idx]
        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else:     # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            if Area_square < 0:  # negative due to floating point precision
                h = 0
                x = d_ss + d1
            else:
                Area = Area_square**0.5
                h = Area * 2/d_ss
                x = (d1**2 - h**2)**0.5

        return x, h

class PlotLocalMap(LocalMap):
    def __init__(self, track):
        super().__init__(track)

        # self.local_map_img_path = self.path + "LocalMapImgs/"
        # ensure_path_exists(self.local_map_img_path)
    
    def plot_local_map(self, save_path=None, counter=0, xs=None, ys=None):
        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]

        plt.figure(1)
        plt.clf()
        if xs is not None and ys is not None:
            plt.plot(xs, ys, '.', color='#0057e7', alpha=0.7)
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

        if save_path is not None:
            plt.savefig(save_path + f"Local_map_{counter}.svg")

    def plot_local_map_offset(self, offset_pos, offset_theta, origin, resolution, save_path=None, counter=0):
        l1 = self.track[:, :2] + self.nvecs * self.track[:, 2][:, None]
        l2 = self.track[:, :2] - self.nvecs * self.track[:, 3][:, None]

        rotation = np.array([[np.cos(offset_theta), -np.sin(offset_theta)],
                                [np.sin(offset_theta), np.cos(offset_theta)]])
        
        l1 = np.matmul(rotation, l1.T).T
        l2 = np.matmul(rotation, l2.T).T

        l1 = l1 + offset_pos
        l2 = l2 + offset_pos

        l1 = (l1 - origin) / resolution
        l2 = (l2 - origin) / resolution

        track = np.matmul(rotation, self.track[:, :2].T).T
        track = track + offset_pos
        track = (track - origin) / resolution

        position = (offset_pos - origin) / resolution

        # plt.figure(1)
        # plt.clf()
        plt.plot(track[:, 0], track[:, 1], '--', color='#E74C3C', label="Center", linewidth=2)
        plt.plot(track[0, 0], track[0, 1], '*', color='#E74C3C', markersize=10)

        plt.plot(l1[:, 0], l1[:, 1], color='#ffa700')
        plt.plot(l2[:, 0], l2[:, 1], color='#ffa700')

        plt.gca().set_aspect('equal', adjustable='box')

        buffer = 50

        plt.xlim([np.min(track[:, 0]) - buffer, np.max(track[:, 0]) + buffer])
        plt.ylim([np.min(track[:, 1]) - buffer, np.max(track[:, 1]) + buffer])

        if save_path is not None:
            plt.savefig(save_path + f"Local_map_{counter}.svg")



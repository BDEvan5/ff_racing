import numpy as np
import matplotlib.pyplot as plt
import LocalMapRacing.tph_utils as tph
from matplotlib.collections import LineCollection
np.set_printoptions(precision=4)
from LocalMapRacing.local_mapping.local_map_utils import *
from LocalMapRacing.local_mapping.LocalMap import LocalMap, PlotLocalMap


DISTNACE_THRESHOLD = 1.4 # distance in m for an exception
TRACK_WIDTH = 1.9 # use fixed width
POINT_SEP_DISTANCE = 0.8


class LocalMapGenerator:
    def __init__(self, path, map_name) -> None:
        fov2 = 4.7 / 2
        self.angles = np.linspace(-fov2, fov2, 1080)
        self.coses = np.cos(self.angles)
        self.sines = np.sin(self.angles)
        self.xs, self.ys = None, None 

        self.local_map_data_path = path + f"LocalMapData_{map_name.upper()}/"
        ensure_path_exists(self.local_map_data_path)
        self.counter = 0

    def generate_line_local_map(self, scan, save=True):
        xs_f = self.coses * scan
        ys_f = self.sines * scan

        pts, pt_distances, inds = self.extract_track_lines(xs_f, ys_f)
        long_side, short_side = self.extract_boundaries(pts, pt_distances, inds)
        track = self.project_side_to_track(long_side)

        local_map = PlotLocalMap(track)
        # lm = LocalMap(track)

        if save: np.save(self.local_map_data_path + f"local_map_{self.counter}", local_map.track)
        self.counter += 1

        return local_map
    
    def extract_track_lines(self, xs, ys):
        pts = np.hstack((xs[:, None], ys[:, None]))
        pts = pts[pts[:, 0] > -2] # remove points behind the car
        pts = pts[np.logical_or(pts[:, 0] > 0, np.abs(pts[:, 1]) < 2)] # remove points behind the car or too far away
        pt_distances = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        inds = np.array(np.where(pt_distances > DISTNACE_THRESHOLD))
        exclusion_zone = 2
        length = len(pts)
        inds = np.delete(inds, np.where(inds <= exclusion_zone)) 
        inds = np.delete(inds, np.where(inds >= length-exclusion_zone)) 

        if len(inds) == 0:
            raise IOError("Problem with full scan, no gaps found")

        return pts, pt_distances, inds

    def extract_boundaries(self, pts, pt_distances, inds):
        arr_inds = np.arange(len(pt_distances))[inds]
        min_ind = np.min(arr_inds) + 1
        max_ind = np.max(arr_inds) + 1

        l1_cs = np.cumsum(pt_distances[:min_ind-1])
        l2_cs = np.cumsum(pt_distances[max_ind:])

        if l1_cs[-1] > l2_cs[-1]:
            long_pts = pts[:min_ind]
            long_length = l1_cs[-1]

            short_pts = pts[max_ind:]
            short_pts = short_pts[::-1]
            short_length = l2_cs[-1]
        else:
            long_pts = pts[max_ind:]
            long_length = l2_cs[-1]

            short_pts = pts[:min_ind]
            short_length = l1_cs[-1]

        n_pts = int(short_length / POINT_SEP_DISTANCE)
        short_side = interpolate_track(short_pts, n_pts*2, 0)

        n_pts = int(long_length / POINT_SEP_DISTANCE)
        long_side = interpolate_track(long_pts, n_pts*2, 0)

        return long_side, short_side

    def project_side_to_track(self, side):
        side_lm = LocalMap(side)
        center_line = side + side_lm.nvecs * TRACK_WIDTH / 2
        n_pts = int(side.shape[0] / 2)
        center_line = interpolate_track(center_line, n_pts, 0)

        center_line = center_line[center_line[:, 0] > 0] # remove points behind the car

        pt_init = np.linalg.norm(center_line[0, :])
        pt_final = np.linalg.norm(center_line[-1, :])
        if pt_final < pt_init: center_line = np.flip(center_line, axis=0)

        ws = np.ones_like(center_line) * TRACK_WIDTH / 2
        track = np.concatenate((center_line, ws), axis=1)

        return track

    # def adjust_track_normals(self, lm):
    #     crossing_horizon = min(5, len(lm.track)//2 -1)
    #     i = 0
    #     while i < 20 and tph.check_normals_crossing.check_normals_crossing(lm.track, lm.nvecs, crossing_horizon):
    #         i += 1
    #         if np.mean(lm.kappa) > 0:
    #             lm.track[:, 2] *= 0.9
    #         else:
    #             lm.track[:, 3] *= 0.9
    #         lm.calculate_length_heading_nvecs()
    #         # print(f"{i}:: Normals crossed --> New width: {lm.track[0, 2:]}")

    #     return lm

        
        



if __name__ == "__main__":
    pass
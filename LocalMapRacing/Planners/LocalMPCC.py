import numpy as np 
import matplotlib.pyplot as plt
import casadi as ca
from matplotlib.collections import LineCollection

from LocalMapRacing.planner_utils.TrackLine import TrackLine
from LocalMapRacing.planner_utils.LocalReference import LocalReference
from LocalMapRacing.planner_utils.VehicleStateHistory import VehicleStateHistory

from LocalMapRacing.local_mapping.LocalMap import LocalMap
from LocalMapRacing.local_mapping.LocalMapGenerator import LocalMapGenerator
from LocalMapRacing.local_mapping.local_map_utils import *



VERBOSE = False
# VERBOSE = True

WAIT_FOR_USER = False
# WAIT_FOR_USER = True

L = 0.33
GRAVITY = 9.81 
MASS = 3.71
MU = 0.8
F_MAX = 1 * GRAVITY * MASS * MU
MAX_ACCELERATION = 8


WEIGHT_PROGRESS = 10
WEIGHT_LAG = 200
WEIGHT_CONTOUR = 1
WEIGHT_STEER = 500
WEIGHT_STEER_CHANGE = 1000
WEIGHT_SPEED_CHANGE = 10

np.printoptions(precision=2, suppress=True)


def normalise_psi(psi):
    while psi > np.pi:
        psi -= 2*np.pi
    while psi < -np.pi:
        psi += 2*np.pi
    return psi


class LocalMPCC:
    def __init__(self, map_name, test_name):
        self.name = test_name
        self.path = f"Data/{test_name}/"
        ensure_path_exists(self.path)
        self.traj_path = f"Data/{test_name}/Trajectories/"
        ensure_path_exists(self.traj_path)

        self.dt = 0.1
        self.N = 10 # number of steps to predict
        self.nx = 4
        self.nu = 3
        self.p_initial = 5
        self.track = TrackLine(map_name, False)
        self.rt = TrackLine(map_name, True)
        self.vehicle_state_history = VehicleStateHistory(test_name, map_name)
        self.counter = 0
        
        self.x_min, self.y_min = -25, -25
        self.psi_min, self.s_min = -100, 0
        self.x_max, self.y_max = 25, 25
        self.psi_max, self.s_max = 100, 20

        self.delta_min, self.p_min, self.v_min = -0.4, 1, 3
        self.delta_max, self.p_max, self.v_max = 0.4, 10, 8
        self.max_v_dot = self.dt * MAX_ACCELERATION

        self.u0 = np.zeros((self.N, self.nu))
        self.X0 = np.zeros((self.N + 1, self.nx))
        self.warm_start = True # warm start every time

        self.init_optimisation()
        self.init_constraints()
        # self.init_objective()
        # self.init_solver()
        
        self.local_map_generator = LocalMapGenerator(self.path)
        self.local_map = None

    def init_optimisation(self):
        # States
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        psi = ca.MX.sym('psi')
        s = ca.MX.sym('s')
        # Controls
        delta = ca.MX.sym('delta')
        v = ca.MX.sym('v')
        p = ca.MX.sym('p')

        states = ca.vertcat(x, y, psi, s)
        controls = ca.vertcat(delta, v, p)
        rhs = ca.vertcat(v * ca.cos(psi), v * ca.sin(psi), (v / L) * ca.tan(delta), p)  # dynamic equations of the states
        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        self.U = ca.MX.sym('U', self.nu, self.N)
        self.X = ca.MX.sym('X', self.nx, (self.N + 1))
        self.P = ca.MX.sym('P', self.nx + 2 * self.N + 1) # init state and boundaries of the reference path

    def init_constraints(self):
        '''Initialize constraints for states, dynamic model state transitions and control inputs of the system'''
        self.lbg = np.zeros((self.nx * (self.N + 1) + self.N*3 , 1))
        self.ubg = np.zeros((self.nx * (self.N + 1) + self.N*3 , 1))
        self.lbx = np.zeros((self.nx + (self.nx + self.nu) * self.N, 1))
        self.ubx = np.zeros((self.nx + (self.nx + self.nu) * self.N, 1))
        # Upper and lower bounds for the state optimization variables
        lbx = np.array([[self.x_min, self.y_min, self.psi_min, self.s_min]])
        ubx = np.array([[self.x_max, self.y_max, self.psi_max, self.s_max]])
        for k in range(self.N + 1):
            self.lbx[self.nx * k:self.nx * (k + 1), 0] = lbx
            self.ubx[self.nx * k:self.nx * (k + 1), 0] = ubx
        state_count = self.nx * (self.N + 1)
        # Upper and lower bounds for the control optimization variables
        for k in range(self.N):
            self.lbx[state_count:state_count + self.nu, 0] = np.array([[self.delta_min, self.v_min, self.p_min]])  # v and delta lower bound
            self.ubx[state_count:state_count + self.nu, 0] = np.array([[self.delta_max, self.v_max, self.p_max]])  # v and delta upper bound
            state_count += self.nu

    def init_objective(self, rp):
        self.obj = 0  # Objective function
        self.g = []  # constraints vector

        st = self.X[:, 0]  # initial state
        self.g = ca.vertcat(self.g, st - self.P[:self.nx])  # initial condition constraints
        for k in range(self.N):
            st = self.X[:, k]
            st_next = self.X[:, k + 1]
            con = self.U[:, k]
            
            t_angle = rp.angle_lut_t(st_next[3])
            ref_x, ref_y = rp.center_lut_x(st_next[3]), rp.center_lut_y(st_next[3])
            #Contouring error
            e_c = ca.sin(t_angle) * (st_next[0] - ref_x) - ca.cos(t_angle) * (st_next[1] - ref_y)
            #Lag error
            e_l = -ca.cos(t_angle) * (st_next[0] - ref_x) - ca.sin(t_angle) * (st_next[1] - ref_y)

            self.obj = self.obj + e_c **2 * WEIGHT_CONTOUR  
            self.obj = self.obj + e_l **2 * WEIGHT_LAG
            self.obj = self.obj - con[2] * WEIGHT_PROGRESS 
            self.obj = self.obj + (con[0]) ** 2 * WEIGHT_STEER  # minimize the use of steering input

            k1 = self.f(st, con)
            st_next_euler = st + (self.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # compute constraints

            # path boundary constraints
            self.g = ca.vertcat(self.g, self.P[self.nx + 2 * k] * st_next[0] - self.P[self.nx + 2 * k + 1] * st_next[1])  # LB<=ax-by<=UB  --represents half space planes

            force_lateral = con[1] **2 / L * ca.tan(ca.fabs(con[0])) *  MASS
            self.g = ca.vertcat(self.g, force_lateral)

            if k == 0: 
                self.g = ca.vertcat(self.g, ca.fabs(con[1] - self.P[-1]))
            else:
                self.obj = self.obj + (con[0] - self.U[0, k - 1]) ** 2 * WEIGHT_STEER_CHANGE  # minimize the change of steering input
                self.obj = self.obj + (con[1] - self.U[1, k - 1]) ** 2 * WEIGHT_SPEED_CHANGE  # minimize the change of steering input
                self.g = ca.vertcat(self.g, ca.fabs(con[1] - self.U[1, k - 1]))  # prevent velocity change from being too abrupt
        
        # if VERBOSE:
        #     print(f"G shape: {self.g.shape}")

    def init_solver(self):
        opts = {}
        opts["ipopt"] = {}
        opts["ipopt"]["max_iter"] = 1000
        opts["jit"] = True
        opts["ipopt"]["acceptable_tol"] = 1e-8
        opts["ipopt"]["acceptable_obj_change_tol"] = 1e-6
        opts["ipopt"]["fixed_variable_treatment"] = "make_parameter"
        opts["ipopt"]["print_level"] = 0
        opts["print_time"] = 0
        
        OPT_variables = ca.vertcat(ca.reshape(self.X, self.nx * (self.N + 1), 1),
                                ca.reshape(self.U, self.nu * self.N, 1))

        nlp_prob = {'f': self.obj, 'x': OPT_variables, 'g': self.g, 'p': self.P}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        
    def plan(self, obs):
        self.local_map = self.local_map_generator.generate_line_local_map(obs['scans'][0])

        rp = LocalReference(self.local_map)

        x0 = np.zeros(3)
        x0 = np.append(x0, rp.calculate_s(x0[0:2]))

        self.init_objective(rp)
        self.init_solver()

        if self.warm_start:
            self.construct_warm_start_soln(x0, rp) 

        # self.track.plot_vehicle([obs['poses_x'][0], obs['poses_y'][0]], obs['poses_theta'][0])
        fs = obs['full_states'][0]
        p = self.generate_constraints_and_parameters(x0, fs[3], rp)
        states, controls, solved_status = self.solve(p)
        if not solved_status:
            print("MPC not solved")
            self.construct_warm_start_soln(x0, rp) 
            states, controls, solved_status = self.solve(p)
            if not solved_status:
                print("MPC still not solved ------------->")

        np.save(self.traj_path + f"States_{self.counter}.npy", states)
        np.save(self.traj_path + f"Controls_{self.counter}.npy", controls)
        first_control = controls[0, :]
        action = first_control[0:2]

        progress = self.track.calculate_progress_percent([obs['poses_x'][0], obs['poses_y'][0]]) * 100
        print(f"S:{progress:2f} --> Action: {action}")

        if VERBOSE:
            rp.plot_path()
            plt.figure(2)
            points = states[:, 0:2].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, 8)
            lc = LineCollection(segments, cmap='jet', norm=norm)
            lc.set_array(controls[:, 1])
            lc.set_linewidth(4)
            line = plt.gca().add_collection(lc)
            plt.colorbar(line, fraction=0.046, pad=0.04, shrink=0.99)

            size = 10
            plt.xlim([-1, x0[0] + size])
            plt.ylim([-5, 5])
            # plt.ylim([x0[1] - size, x0[1] + size])
            plt.pause(0.001)

        self.vehicle_state_history.add_memory_entry(obs, action)
        self.counter += 1

        # plt.show()

        return action

    def plan_old(self, obs):
        x0 = obs["full_states"][0][np.array([0, 1, 4])]
        x0[2] = normalise_psi(x0[2]) 

        x0 = np.append(x0, self.rp.calculate_s(x0[0:2]))
        x0[3] = self.filter_estimate(x0[3])
        fs = obs["full_states"][0]

        p = self.generate_constraints_and_parameters(x0, fs[3])
        states, controls, solved_status = self.solve(p)
        if not solved_status:
            self.warm_start = True
            p = self.generate_constraints_and_parameters(x0, fs[3])
            states, controls, solved_status = self.solve(p)
            if VERBOSE:
                print(f"Solve failed: ReWarm Start: New outcome: {solved_status}")

        s = states[:, 3]
        s = [s[k] if s[k] < self.rp.track_length else s[k] - self.rp.track_length for k in range(self.N+1)]
        c_pts = [[self.rp.center_lut_x(states[k, 3]).full()[0, 0], self.rp.center_lut_y(states[k, 3]).full()[0, 0]] for k in range(self.N + 1)]

        first_control = controls[0, :]
        action = first_control[0:2]
        if VERBOSE:
            print(f"S:{x0[3]:2f} --> Action: {action}")
            if not solved_status:
                plt.show()

        if VERBOSE:
            self.rp.plot_path()
            plt.figure(2)
            points = states[:, 0:2].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, 8)
            lc = LineCollection(segments, cmap='jet', norm=norm)
            lc.set_array(controls[:, 1])
            lc.set_linewidth(4)
            line = plt.gca().add_collection(lc)
            plt.colorbar(line, fraction=0.046, pad=0.04, shrink=0.99)

            # plt.plot(states[:, 0], states[:, 1], '-o', color='red')
            for i in range(self.N + 1):
                xs = [states[i, 0], c_pts[i][0]]
                ys = [states[i, 1], c_pts[i][1]]
                plt.plot(xs, ys, '--', color='orange')

            size = 12
            plt.xlim([x0[0] - size, x0[0] + size])
            plt.ylim([x0[1] - size, x0[1] + size])
            plt.pause(0.001)

            # if not solved_status:
            #     plt.show()

        if VERBOSE:
            fig, axs = plt.subplots(5, 1, num=3, clear=True, figsize=(8, 15))
            axs[0].plot(controls[:, 0], '-o', color='red')
            axs[0].set_ylabel('Steering Angle')
            axs[0].set_ylim([-0.5, 0.5])
            axs[0].grid(True)
            axs[0].set_title(f"Speed: {fs[3]:2f}")

            axs[1].plot(controls[:, 1], '-o', color='red')
            vs = self.rt.get_interpolated_vs(states[:, 3])
            axs[1].plot(vs, '-o', color='blue')
            axs[1].set_ylabel('Speed')
            axs[1].set_ylim([0, 9])
            axs[1].grid(True)

            forces = [controls[k, 1] **2 / L * ca.tan(ca.fabs(controls[k, 0])) *  MASS for k in range(self.N)]
            axs[2].plot(forces, '-o', color='red')
            axs[2].set_ylabel('Lateral Force')
            axs[2].set_ylim([0, 40])
            axs[2].grid(True)


            axs[3].plot(controls[:, 2], '-o', color='red')
            axs[3].set_ylabel('Centerline \nSpeed')
            axs[3].set_ylim([0, 9])
            axs[3].grid(True)

            dv = np.diff(controls[:, 1])
            dv = np.insert(dv, 0, controls[0, 1]- fs[3])
            # dv = np.insert(0, fs[3] - controls[0, 1])

            axs[4].plot(dv, '-o', color='red')
            axs[4].set_ylabel('Acceleration')
            # axs[4].set_ylim([-2, 2])
            axs[4].grid(True)

            plt.pause(0.001)
            # plt.pause(0.1)

            # if action[1] < 4:
            #     plt.show()

        if VERBOSE and False:
            plt.figure(4)
            plt.clf()
            plt.plot(self.rp.center_lut_x(self.rp.s_track), self.rp.center_lut_y(self.rp.s_track), label="center", color='green', alpha=0.7)
            plt.plot(self.rp.left_lut_x(self.rp.s_track), self.rp.left_lut_y(self.rp.s_track), label="left", color='green', alpha=0.7)
            plt.plot(self.rp.right_lut_x(self.rp.s_track), self.rp.right_lut_y(self.rp.s_track), label="right", color='green', alpha=0.7)

            plt.plot(states[:, 0], states[:, 1], '-o', color='red')
            # plt.plot(c_pts[:, 0], c_pts[:, 1], '-o', color='blue')
            # plt.plot(x0[0], x0[1], 'x', color='green')
            plt.plot(self.rt.wpts[:, 0], self.rt.wpts[:, 1], color='blue')

            size = 12
            plt.xlim([x0[0] - size, x0[0] + size])
            plt.ylim([x0[1] - size, x0[1] + size])
            plt.pause(0.001)


        if VERBOSE and WAIT_FOR_USER:
            plt.show()


        self.vehicle_state_history.add_memory_entry(obs, action)

        return action # return the first control action

    def generate_constraints_and_parameters(self, x0_in, x0_speed, rp):


        p = np.zeros(self.nx + 2 * self.N + 1)
        p[:self.nx] = x0_in

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s = self.X0[k, 3]
            right_point = [rp.right_lut_x(s).full()[0, 0], rp.right_lut_y(s).full()[0, 0]]
            left_point = [rp.left_lut_x(s).full()[0, 0], rp.left_lut_y(s).full()[0, 0]]

            delta_x_path = right_point[0] - left_point[0]
            delta_y_path = right_point[1] - left_point[1]
            p[self.nx + 2 * k:self.nx + 2 * k + 2] = [-delta_x_path, delta_y_path]

            up_bound = max(-delta_x_path * right_point[0] - delta_y_path * right_point[1],
                           -delta_x_path * left_point[0] - delta_y_path * left_point[1])
            low_bound = min(-delta_x_path * right_point[0] - delta_y_path * right_point[1],
                            -delta_x_path * left_point[0] - delta_y_path * left_point[1])
            self.lbg[self.nx - 3 + (self.nx + 3) * (k + 1), 0] = low_bound # check this, there could be an error
            self.ubg[self.nx - 3 + (self.nx + 3) * (k + 1), 0] = up_bound
            self.lbg[self.nx - 2 + (self.nx + 3) * (k + 1), 0] = - F_MAX
            self.ubg[self.nx - 2 + (self.nx + 3) * (k + 1) , 0] = F_MAX
            #Adjust indicies.
            self.lbg[self.nx -1 + (self.nx + 3) * (k + 1), 0] = - self.max_v_dot
            self.ubg[self.nx -1 + (self.nx + 3) * (k + 1) , 0] = ca.inf # do not limit speeding up
            # self.ubg[self.nx -1 + (self.nx + 3) * (k + 1) , 0] = self.max_v_dot


        # the optimizer cannot control the init state.
        self.lbg[self.nx *2, 0] = - ca.inf
        self.ubg[self.nx *2, 0] = ca.inf

        p[-1] = max(x0_speed, 1) # prevent constraint violation

        return p

    def solve(self, p):
        x_init = ca.vertcat(ca.reshape(self.X0.T, self.nx * (self.N + 1), 1),
                         ca.reshape(self.u0.T, self.nu * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        # Get state and control solution
        self.X0 = ca.reshape(sol['x'][0:self.nx * (self.N + 1)], self.nx, self.N + 1).T  # get soln trajectory
        u = ca.reshape(sol['x'][self.nx * (self.N + 1):], self.nu, self.N).T  # get controls solution

        trajectory = self.X0.full()  # size is (N+1,n_states)
        inputs = u.full()
        stats = self.solver.stats()
        solved_status = True
        if stats['return_status'] == 'Infeasible_Problem_Detected':
        # if stats['return_status'] != 'Solve_Succeeded':
            if VERBOSE:
                print(stats['return_status'])
            solved_status = False

        # Shift trajectory and control solution to initialize the next step
        self.X0 = ca.vertcat(self.X0[1:, :], self.X0[self.X0.size1() - 1, :])
        self.u0 = ca.vertcat(u[1:, :], u[u.size1() - 1, :])

        return trajectory, inputs, solved_status
        
    def filter_estimate(self, initial_arc_pos):
        if (self.X0[0, 3] >= self.rp.track_length) and (
                (initial_arc_pos >= self.rp.track_length) or (initial_arc_pos <= 5)):
            self.X0[:, 3] = self.X0[:, 3] - self.rp.track_length
        if initial_arc_pos >= self.rp.track_length:
            initial_arc_pos -= self.rp.track_length
        return initial_arc_pos

    def construct_warm_start_soln(self, initial_state, rp):
        self.X0 = np.zeros((self.N + 1, self.nx))
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + self.p_initial * self.dt

            psi_next = rp.angle_lut_t(s_next).full()[0, 0]
            x_next, y_next = rp.center_lut_x(s_next), rp.center_lut_y(s_next)

            # adjusts the centerline angle to be continuous
            psi_diff = self.X0[k-1, 2] - psi_next
            psi_mul = self.X0[k-1, 2] * psi_next
            if (abs(psi_diff) > np.pi and psi_mul < 0) or abs(psi_diff) > np.pi*1.5:
                if psi_diff > 0:
                    psi_next += np.pi * 2
                else:
                    psi_next -= np.pi * 2

            self.X0[k, :] = np.array([x_next.full()[0, 0], y_next.full()[0, 0], psi_next, s_next])

        self.u0 = np.zeros((self.N, self.nu))
        self.u0[:, 1] = self.p_initial
        self.u0[:, 2] = self.p_initial

        self.warm_start = False

    def done_callback(self, final_obs):
        self.vehicle_state_history.add_memory_entry(final_obs, np.array([0, 0]))
        self.vehicle_state_history.save_history()

        progress = self.track.calculate_progress_percent([final_obs['poses_x'][0], final_obs['poses_y'][0]]) * 100
        
        print(f"Lap complete ({self.track.map_name.upper()}) --> Time: {final_obs['lap_times'][0]:.2f}, Progress: {progress:.1f}%")


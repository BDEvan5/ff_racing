from LocalMapRacing.f1tenth_gym.f110_env import F110Env
from LocalMapRacing.Planners.LocalMPCC import LocalMPCC
from LocalMapRacing.Planners.LocalMapPP import LocalMapPP


import numpy as np

RENDER_ENV = False
# RENDER_ENV = True


        
def run_simulation_loop_laps(env, planner, n_laps, n_sim_steps=10):
    init_positions = np.array([[0, 0, 0]])
    # init_positions = np.array([[2.45, -14.9, 2.7]])

    observation, reward, done, info = env.reset(poses=init_positions)
    
    for lap in range(n_laps):
        while not done:
            action = planner.plan(observation)
            
            mini_i = n_sim_steps
            while mini_i > 0 and not done:
                observation, reward, done, info = env.step(action[None, :])
                mini_i -= 1

            if RENDER_ENV: env.render('human_fast')
            
        planner.done_callback(observation)
        observation, reward, done, info = env.reset(poses=init_positions)
        


def render_callback(env_renderer):
        e = env_renderer
        block_size = 400

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - int(block_size * 0.8)
        e.left = left - block_size
        e.right = right + block_size
        e.top = top + block_size
        e.bottom = bottom - block_size



def test_single_map():
    map_name = "aut" # "aut", "esp", "gbr", "mco"
    # map_name = "esp" # "aut", "esp", "gbr", "mco"
    n_test_laps = 1
    set_n = 1
    # test_name = "DataLocalPP"
    # test_name = "devel_local_mpcc"
    test_name = f"LocalMPCC_{set_n}"
    # test_name = f"LocalCenterPP_{set_n}"
    
    env = F110Env(map=map_name, num_agents=1)
    # planner = LocalMapPP(test_name, map_name)
    planner = LocalMPCC(map_name, test_name)
    
    run_simulation_loop_laps(env, planner, n_test_laps, 10)
  

def test_pure_pursuit_all_maps():
    map_names = ["aut", "esp", "gbr", "mco"]
    # map_names = ["aut"]
    n_test_laps = 1
    
    set_n = 1
    # test_name = f"LocalPP"
    # test_name = f"LocalCenterPP_{set_n}"
    test_name = f"LocalRacePP_{set_n}"
    for map_name in map_names:
        # 
        env = F110Env(map=map_name, num_agents=1)
        planner = LocalMapPP(test_name, map_name)
        
        run_simulation_loop_laps(env, planner, n_test_laps, 10)
        F110Env.renderer = None  

def test_mpcc_all_maps():
    map_names = ["aut", "esp", "gbr", "mco"]
    map_names = ["esp", "gbr", "mco"]
    n_test_laps = 1
    
    set_n = 1
    test_name = f"LocalMPCC_{set_n}"
    for map_name in map_names:
        
        env = F110Env(map=map_name, num_agents=1)
        planner = LocalMPCC(map_name, test_name)
        
        run_simulation_loop_laps(env, planner, n_test_laps, 10)
        F110Env.renderer = None
  

if __name__ == "__main__":
    # test_single_map()
    # test_pure_pursuit_all_maps()
    test_mpcc_all_maps()






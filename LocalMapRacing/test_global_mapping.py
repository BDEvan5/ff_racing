from LocalMapRacing.f1tenth_gym.f110_env import F110Env
from LocalMapRacing.Planners.GlobalMPCC import GlobalMPCC
from LocalMapRacing.Planners.PurePursuit import PurePursuit


import numpy as np

RENDER_ENV = False
# RENDER_ENV = True


        
def run_simulation_loop_laps(env, planner, n_laps, n_sim_steps=10):
    observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))
    
    for lap in range(n_laps):
        while not done:
            action = planner.plan(observation)
            
            mini_i = n_sim_steps
            while mini_i > 0 and not done:
                observation, reward, done, info = env.step(action[None, :])
                mini_i -= 1

            if RENDER_ENV: env.render('human_fast')
            
        planner.done_callback(observation)
        observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))   
        

def test_pure_pursuit():
    map_name = "aut" # "aut", "esp", "gbr", "mco"
    n_test_laps = 1
    test_name = "time_pp"

    
    env = F110Env(map=map_name, num_agents=1)
    # planner = GlobalMPCC(map_name, test_name)
    planner = PurePursuit(map_name, test_name)
    
    run_simulation_loop_laps(env, planner, n_test_laps, 1)
  
def test_pure_pursuit_all_maps():
    map_names = ["aut", "esp", "gbr", "mco"]
    n_test_laps = 1
    
    set_n = 1
    test_name = f"GlobalPP"
    for map_name in map_names:
        
        env = F110Env(map=map_name, num_agents=1)
        planner = PurePursuit(map_name, test_name)
        
        run_simulation_loop_laps(env, planner, n_test_laps, 1)
        F110Env.renderer = None  

def test_mpcc_all_maps():
    map_names = ["aut", "esp", "gbr", "mco"]
    n_test_laps = 1
    
    set_n = 1
    test_name = f"GlobalMPCC"
    for map_name in map_names:
        
        env = F110Env(map=map_name, num_agents=1)
        planner = GlobalMPCC(map_name, test_name)
        
        run_simulation_loop_laps(env, planner, n_test_laps, 10)
        F110Env.renderer = None
  
def run_profiling(function, name):
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    function()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    with open(f"Data/profile_{name}.txt", "w") as f:
        ps.print_stats()
        f.write(s.getvalue())

if __name__ == "__main__":
    # test_pure_pursuit()
    run_profiling(test_pure_pursuit, "GlobalAUT")
    # test_pure_pursuit_all_maps()
    # test_mpcc_all_maps()
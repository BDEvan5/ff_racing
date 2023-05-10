from ff_racing.f1tenth_gym.f110_env import F110Env

from ff_racing.Planners.FrenetFramePlanner import FrenetFramePlanner

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

            # if RENDER_ENV: env.render('human')
            if RENDER_ENV: env.render('human_fast')
            
        planner.done_callback(observation)
        observation, reward, done, info = env.reset(poses=np.array([[0, 0, 0]]))   
        

def test_frenet_planner():
    map_name = "aut" # "aut", "esp", "gbr", "mco"
    agent_name = "devel_ff"
    n_test_laps = 1
    
    env = F110Env(map=map_name, num_agents=1)
    
    agent_name = "MyFrenetPlanner"
    planner = FrenetFramePlanner(agent_name, f"Data/{agent_name}/")
    run_simulation_loop_laps(env, planner, n_test_laps, 10)
  
  
  
def test_endToEnd_agent_all_maps():
    map_names = ["aut", "esp", "gbr", "mco"]
    agent_name = "myFavouriteAgent_SAC"
    n_test_laps = 2
    
    for map_name in map_names:
        env = F110Env(map=map_name, num_agents=1)
        TestAgent = TestSAC(agent_name, f"Data/{agent_name}/") # or DDPG, TD3
        planner = EndToEndTest(TestAgent, map_name, agent_name)
        run_simulation_loop_laps(env, planner, n_test_laps)
  
  
if __name__ == "__main__":
    test_frenet_planner()
from ff_racing.f1tenth_gym.f110_env import F110Env

from ff_racing.Planners.FrenetFramePlanner import FrenetFramePlanner
from ff_racing.Planners.LocalMapPlanner import LocalMapPlanner

import numpy as np

RENDER_ENV = False
RENDER_ENV = True


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


def test_frenet_planner():
    map_name = "aut" # "aut", "esp", "gbr", "mco"
    # map_name = "mco"
    n_test_laps = 1
    
    env = F110Env(map=map_name, num_agents=1)
    env.add_render_callback(render_callback)

    set_n = 1
    agent_name = f"LocalMapPlanner_{set_n}"
    planner = LocalMapPlanner(agent_name, f"Data/{agent_name}/", map_name)
    run_simulation_loop_laps(env, planner, n_test_laps, 10)
  
def test_lm_planner_all():
    map_list = ["aut", "esp", "gbr", "mco"]
    n_test_laps = 1
    set_n = 1
    agent_name = f"LocalMapPlanner_{set_n}"

    for map_name in map_list:
        env = F110Env(map=map_name, num_agents=1)
        env.add_render_callback(render_callback)

        planner = LocalMapPlanner(agent_name, f"Data/{agent_name}/", map_name)
        run_simulation_loop_laps(env, planner, n_test_laps, 10)
  
  
if __name__ == "__main__":
    # test_frenet_planner()
    test_lm_planner_all()
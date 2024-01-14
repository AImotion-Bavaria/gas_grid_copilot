"""
This file deals with exploring a space of policies.

"""
from simple_gym_pandapipes import SimpleGasStorageEnv, FixedDummyAgent, run_trajectory, train_SB_Agent, simulate_trained_policy
from simple_storage import get_example_line
from stable_baselines3 import PPO, SAC
from viz4 import IKIGasDemoPlot
import numpy as np


def prepare_trained_models(env):
    steps = [1., 10., 100.]
    plot_data_dict = dict() 
    for r1 in steps:
        for r2 in steps:
            for r3 in steps:
                env.reward_weights = np.array([r1, r2, r3])
                env.reward_weights /= np.sum(env.reward_weights)
                model_file_name = f"cached_models/SAC_{int(r1)}_{int(r2)}_{int(r3)}.zip"
                agent = train_SB_Agent(env, algorithm=SAC, force_retraining=False, total_timesteps= 10000, model_file_name = model_file_name)

                rewards, imgs = simulate_trained_policy(env, agent)
                plot_data_dict[(int(r1), int(r2), int(r3))] = (imgs, env.get_output_dict(), rewards)
    return plot_data_dict

import matplotlib.pyplot as plt
import os 
import pickle
 
if __name__ == "__main__":
    env = SimpleGasStorageEnv(get_example_line, normalize_rewards=True)
    obs, info = env.reset()
    print(obs)

    # just a dummy agent alway wanting the maximal inflow
    # just dummy rotating inflows
    inflows = [SimpleGasStorageEnv.MAX_STORAGE_MDOT_KG_PER_S, 0., SimpleGasStorageEnv.MIN_STORAGE_MDOT_KG_PER_S/2]
    fixed_dummy = FixedDummyAgent(inflows)
    #run_trajectory(env, fixed_dummy)
    plot_data = prepare_trained_models(env)
    
    plot_data_name = f"cached_models/prepared_plot_data.pickle"
    plot_data_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), plot_data_name)
    # # using with statement
    with open(plot_data_name, 'wb') as file:
         pickle.dump(plot_data, file)

    #rewards, imgs = run_trajectory(env, trained_agent)
    # load prepared data 
    with open(plot_data_name, 'rb') as file:
         plot_data = pickle.load(file)
        
    #imgs, obs_dict, rewards = plot_data[(1,1,1)]

    #demo_plot = IKIGasDemoPlot()
    #demo_plot.image_plot(imgs, obs_dict, rewards, plot_data)
    #plt.show()
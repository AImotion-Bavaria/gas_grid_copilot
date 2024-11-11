"""
This file deals with exploring a space of policies. It allows us to train 
a set of models with different weightings for the respective rewards.

The trained models are then saved to the folder `cached_models` and, once trained,
the IKIGas dashboard is loaded.

"""
from simple_gym_pandapipes import SimpleGasStorageEnv, FixedDummyAgent, run_trajectory, train_SB_Agent, simulate_trained_policy
from simple_storage import get_example_line
from stable_baselines3 import PPO, SAC
from ikigas_dashboard import IKIGasDashboard
import blosc
import numpy as np
import pandas as pd
import seaborn as sns 

import matplotlib.pyplot as plt
import os 
import pickle

def prepare_trained_models(env, algo=SAC):
    """ intended for overnight use
    trains and stores all 27 models (= 3*3*3 for 1, 10, 100), i.e., for a given priorization
    of reward1, reward2, reward2, run a fixed training and store the model

    Args:
        env (PandapipesGasNet): 
        algo (_type_, optional): StableBaselines3 RL algorithm, currently we only support SAC and PPO. Defaults to SAC for q-values.

    Returns:
        _type_: _description_
    """
    reward_weight_steps = [1., 10., 100.]
    plot_data_dict = dict() 
    for rw1 in reward_weight_steps:
        for rw2 in reward_weight_steps:
            for rw3 in reward_weight_steps:
                env.reward_weights = np.array([rw1, rw2, rw3])
                env.reward_weights /= np.sum(env.reward_weights)
                model_file_name = f"cached_models/{algo.__name__}_{int(rw1)}_{int(rw2)}_{int(rw3)}.zip"
                agent = train_SB_Agent(env, algorithm=algo, force_retraining=False, total_timesteps= 10000, model_file_name = model_file_name)

                rewards, imgs, q_vals = simulate_trained_policy(env, agent)
                plot_data_dict[(int(rw1), int(rw2), int(rw3))] = (imgs, env.get_output_dict(), rewards, q_vals)
    return plot_data_dict

def combine_reward_trajectories(plot_data):
    # essentially just build a pandas data frame with all the rewards as columns and the states as rows
    reward_dfs = []
    for imgs, obs_dict, rewards, q_vals in plot_data.values():
        reward_dfs.append(rewards)
    reward_df = pd.concat( reward_dfs, ignore_index=True)
    return reward_df
 
if __name__ == "__main__":
    env = SimpleGasStorageEnv(get_example_line, normalize_rewards=True)
    obs, info = env.reset()
    print(obs)

    prepare_and_train = False
    if prepare_and_train:
        algo = SAC
        # just a dummy agent alway wanting the maximal inflow
        # just dummy rotating inflows
        inflows = [SimpleGasStorageEnv.MAX_STORAGE_MDOT_KG_PER_S, 0., SimpleGasStorageEnv.MIN_STORAGE_MDOT_KG_PER_S/2]
        fixed_dummy = FixedDummyAgent(inflows)
        #run_trajectory(env, fixed_dummy)
        for algo in [PPO, SAC]: 
            plot_data = prepare_trained_models(env, algo)
        
            plot_data_name = f"cached_models/{algo.__name__}_prepared_plot_data.pickle"
            plot_data_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), plot_data_name)
            # # using with statement
            pickled_data = pickle.dumps(plot_data)  # returns data as a bytes object
            compressed_pickle = blosc.compress(pickled_data)

            with open(plot_data_name, 'wb') as file:
                #pickle.dump(plot_data, file)
                file.write(compressed_pickle)
    else:
        # load prepared data and start visualization
        algo = SAC
 
        plot_data_name = f"cached_models/{algo.__name__}_prepared_plot_data.pickle"
        import os
        plot_data_name = os.path.join(os.path.dirname(__file__), plot_data_name)
        with open(plot_data_name, 'rb') as file:
            compressed_pickle = file.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        plot_data = pickle.loads(depressed_pickle)  # turn bytes object back into data

        # calculate interaction matrix using plot_data
        all_reward_trajectories = combine_reward_trajectories(plot_data)
 
        imgs, obs_dict, rewards, q_vals = plot_data[(1,1,1)]

        demo_plot = IKIGasDashboard(use_qualicision=True)
        demo_plot.image_plot(imgs, obs_dict, rewards, plot_data=plot_data, q_vals=q_vals, all_reward_trajectories=all_reward_trajectories)
        plt.show()

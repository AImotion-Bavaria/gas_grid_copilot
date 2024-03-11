import matplotlib.pyplot as plt
import os 
import pickle
import blosc
import pandas as pd
import seaborn as sns 

def combine_reward_trajectories(plot_data):
    # essentially just build a pandas data frame with all the rewards as columns and the states as rows
    reward_dfs = []
    for imgs, obs_dict, rewards, q_vals in plot_data.values():
        reward_dfs.append(rewards)
    reward_df = pd.concat( reward_dfs, ignore_index=True)
    return reward_df

if __name__ == "__main__":
    plot_data_name = f"PPO_prepared_plot_data.pickle"

    plot_data_name = os.path.join(os.path.dirname(__file__), plot_data_name)
    with open(plot_data_name, 'rb') as file:
        compressed_pickle = file.read()
    depressed_pickle = blosc.decompress(compressed_pickle)
    plot_data = pickle.loads(depressed_pickle)  

    # calculate interaction matrix using plot_data
    all_reward_trajectories = combine_reward_trajectories(plot_data)

    # can be any value in [1, 10, 100]
    weight_reward_storage = 1
    weight_reward_mass_flow = 1
    weight_reward_difference = 1

    imgs, obs_dict, rewards, q_vals = plot_data[(weight_reward_storage , weight_reward_mass_flow, weight_reward_difference)]
    # imgs is a set of PIL images for showing the environment
    # obs_dict is an internal data structure of pandapipes
    # rewards is the dataframe containing the rewards per time step
    # q_vals is the sequence of q values the agent reported for the giving weighting
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Simple visualizations')
    interaction_matrix = all_reward_trajectories
    corr = interaction_matrix.corr()
    ax = sns.heatmap(corr, 
        xticklabels=corr.columns.values,
        yticklabels=corr.columns.values,
        cmap=sns.diverging_palette(10, 133, n=256, as_cmap=True),
        ax=ax1)
    plt.tight_layout()
    rewards.plot(ax=ax2)
    plt.show()

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pandapipes as pp 
import pandas as pd
from simple_storage import get_example_line, plot_trajectory, get_multimodal_flow
from storage_controller import StorageController

import pandapipes as pp
import pandas as pd

import pandapower.timeseries as ts
import pandapower.control as control
from pandapower.timeseries import DFData
import numpy as np 
import os 
# importing a grid from the library
from pandapipes.networks import gas_meshed_square

from timeseries_wrapper import TimeseriesWrapper

from stable_baselines3 import PPO, SAC

def storage_reward(x):
    if 0 <= x < 0.25:
        return 0
    elif 0.25 <= x <= 0.5:
        return 4 * (x - 0.25)
    elif 0.5 < x < 0.75:
        return 1 - 4 * (x - 0.5)
    elif 0.75 <= x <= 1:
        return 0

def plot_storage_reward():
    x = np.linspace(0, 1, 1000)
    y = [storage_reward(i) for i in x]

    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of the defined function over the interval [0, 1]')
    plt.grid(True)
    plt.show()

class SimpleGasStorageEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["console"]}
    MIN_STORAGE_MDOT_KG_PER_S = -0.05
    MAX_STORAGE_MDOT_KG_PER_S = 0.05
    MAX_TIME_STEPS = 10

    def __init__(self, net_gen_func, mass_storage_id = 0):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # define action and observation space for gym.spaces  .astype(np.float32)  dtype=np.float32
        self.net_gen_func = net_gen_func
        net = net_gen_func() # just for parameter transfer purposes
        self._verbose = True
        self.current_timestep = 0 # needed to identify the inputs in the time series simulation in pandapipes
        self.min_storage_mdot_kg_per_s = SimpleGasStorageEnv.MIN_STORAGE_MDOT_KG_PER_S
        self.max_storage_mdot_kg_per_s = SimpleGasStorageEnv.MAX_STORAGE_MDOT_KG_PER_S
        self.mass_storage_id = mass_storage_id
        self.action_space = spaces.Box(low=np.array([self.min_storage_mdot_kg_per_s]),
                                       high=np.array([self.max_storage_mdot_kg_per_s]),
                                       shape=(1,),
                                       dtype=np.float32)
        # observing 
        # - the current mass storage level 
        # - the current source inflow
        # - the current sink outflow
        # - the last action we took


        mass_storage = net.mass_storage.loc[mass_storage_id]
        self.min_m_stored_kg, self.max_m_stored_kg = mass_storage["min_m_stored_kg"],  mass_storage["max_m_stored_kg"]
        observations = {
            "mass_storage" : (self.min_m_stored_kg, self.max_m_stored_kg),
            "source_inflow" : (-1.0, 1.0),
            "sink_outflow" : (-1.0, 1.0),
            "last_mass_flow" : (self.min_storage_mdot_kg_per_s, self.max_storage_mdot_kg_per_s),
            "time" : (0, SimpleGasStorageEnv.MAX_TIME_STEPS)
        }
        self.observation_space = self.__convert_to_box(observations)
        self.last_action = mass_storage["mdot_kg_per_s"] 

    def __convert_to_box(self, observations_dict):
        low = []
        high = []

        for key, value in observations_dict.items():
            low_, high_ = value
            low.append(low_)
            high.append(high_)
        return spaces.Box(low=np.array(low),
                   high=np.array(high),
                   shape=(len(observations_dict.keys()),),
                   dtype=np.float32)
        
    def __init_pandapipes_ts(self):
        """ sets up everything we need to run pandapipes time series simulations
        """
        # -> that is what I would actually like to see controlled
        storage_flow_df = pd.DataFrame([0.] * SimpleGasStorageEnv.MAX_TIME_STEPS, columns=['mdot_storage'])
        datasource_storage_flow = DFData(storage_flow_df)    # creating storage unit in the grid, which will be controlled by our controller

        # creating an Object of our new build storage controller, controlling the storage unit
        # THIS is what we control by means of our action!
        ctrl = StorageController(net=self.net, sid=0, data_source=datasource_storage_flow, mdot_profile='mdot_storage')
        self.storage_flow_df = storage_flow_df

        _, source_flow = get_multimodal_flow(SimpleGasStorageEnv.MAX_TIME_STEPS, [2, 6], 0.04)
        source_in_flow_df = pd.DataFrame(source_flow, columns = ["mdot_kg_per_s"])
        const_source = control.ConstControl(self.net, element='source', variable='mdot_kg_per_s',
                                        element_index=self.net.source.index.values,
                                        data_source=DFData(source_in_flow_df),
                                        profile_name="mdot_kg_per_s")
        
        # a bit delayed will be the consumption
        _, sink_flow = get_multimodal_flow(SimpleGasStorageEnv.MAX_TIME_STEPS, [4, 8], 0.03)
        sink_out_flow_df = pd.DataFrame(sink_flow, columns = ["mdot_kg_per_s"])
        const_sink = control.ConstControl(self.net, element='sink', variable='mdot_kg_per_s',
                                        element_index=self.net.sink.index.values,
                                        data_source=DFData(sink_out_flow_df),
                                        profile_name="mdot_kg_per_s")
        
        # defining an OutputWriter to track certain variables
        log_variables = [("mass_storage", "mdot_kg_per_s"), ("res_mass_storage", "mdot_kg_per_s"),
                         ("mass_storage", "m_stored_kg"), ("mass_storage", "filling_level_percent"),
                         ("res_ext_grid", "mdot_kg_per_s"), ("res_source", "mdot_kg_per_s"), ("res_sink", "mdot_kg_per_s")] 
        
        self.ow = ts.OutputWriter(self.net, log_variables=log_variables, output_path=None)
        self.timeseries_wrapper = TimeseriesWrapper(self.net, self.ow, log_variables, self._verbose)

    def __get_obs(self, init=True):
        """
            observations = {
            "mass_storage" : (self.min_m_stored_kg, self.max_m_stored_kg),
            "source_inflow" : (-1.0, 1.0),
            "sink_outflow" : (-1.0, 1.0),
            "last_mass_flow" : (self.min_storage_mdot_kg_per_s, self.max_storage_mdot_kg_per_s),
            "time" : (0, SimpleGasStorageEnv.MAX_TIME_STEPS)
        }
        """
        mass_storage = self.net.mass_storage.loc[self.mass_storage_id]
        if init:
            mass_stored = mass_storage["init_m_stored_kg"] # be careful, if we have already run a single time step
        else: 
            mass_stored = mass_storage["m_stored_kg"] 

        last_flow = self.net.res_mass_storage.loc[self.mass_storage_id]["mdot_kg_per_s"]
        # 0 for now, since we just have one source
        source_mdot_kg_per_s = self.net.res_source.loc[0]["mdot_kg_per_s"]
        sink_mdot_kg_per_s = self.net.res_sink.loc[0]["mdot_kg_per_s"]
        return np.array([mass_stored, source_mdot_kg_per_s, sink_mdot_kg_per_s, last_flow, self.current_timestep ])


    def step(self, action : float):
        # write current action into the data source that the pandapipes simulation is using
        if isinstance(action, np.ndarray): # SB-PPO does this
            action = action.item() 
        self.storage_flow_df.loc[self.current_timestep] = action
        #print(f"* picking action ... {action}")
        # simulating ... 
        self.timeseries_wrapper.run_timestep(self.net, self.current_timestep)

        # evaluating to get the reward ...
        rewards = self.__get_rewards(action)
        # do simple linearization here:
        reward = np.sum([val for key, val in rewards])

        self.current_timestep += 1

        terminated = self.current_timestep >= SimpleGasStorageEnv.MAX_TIME_STEPS
        observation = self.__get_obs(init=False)
        truncated = False
        info = {"rewards" : self.rewards}
        return observation, reward, terminated, truncated, info

    def __get_rewards(self, chosen_action):
        # 1. Storage should be between 25% and 75%
        mass_storage = self.net.mass_storage.loc[self.mass_storage_id]
        filling_level_percent = mass_storage["filling_level_percent"]
        reward_storage = storage_reward(filling_level_percent / 100) # to convert to percentage in [0,1]

        # 2. Minimize external grid mass flow - should neither be positive or negative - ideally just work with charging/discharging storage
        ext_mass_flow = self.net.res_ext_grid.loc[0]["mdot_kg_per_s"]
        reward_mass_flow = 100* -np.abs(ext_mass_flow)

        # 3. Do not change too much   
        difference = abs(self.last_action - chosen_action)
        reward_difference = -difference

        self.rewards = [("reward_storage", reward_storage), ("reward_mass_flow", reward_mass_flow), ("reward_difference", reward_difference)]
        return self.rewards
    
    def reset(self, seed=None, options=None):
        # get all initial parameters set up again
        self.net = self.net_gen_func()
        pp.pipeflow(self.net)
        self.current_timestep = 0
        self.__init_pandapipes_ts()
        info = {}
        return self.__get_obs(), info

    def render(self):
        pass

    def close(self):
        pass

    def get_output_dict(self):
        return self.timeseries_wrapper.output
    
    @property
    def verbose(self):
        """The verbose property to be passed to the time series wrapper."""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        self.timeseries_wrapper.verbose = value

    @verbose.deleter
    def verbose(self):
        del self._verbose

class FixedDummyAgent:
    def __init__(self, actions_to_cycle) -> None:
        self.counter = 0
        self.actions_to_cycle = actions_to_cycle

    def predict(self, obs): 
        action = self.actions_to_cycle[self.counter % len(self.actions_to_cycle)]
        self.counter += 1
        return action, None

def train_SB_Agent(env, algorithm=PPO, force_retraining = False):
    env.reset()
    env.verbose = False
    print("Start training ... ")
    model_file_name = f"{algorithm.__name__}_gas_storage.zip"
    model_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_file_name)

    if (not os.path.exists(model_full_path)) or force_retraining:
        model = algorithm("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        model.save(model_full_path)
    else: 
        model = algorithm.load(model_full_path)
    return model


def run_trajectory(env, agent):
    reward_trajectory = []
    reward_cols = None
    observation, info = env.reset()

    for i in range(0, 10):
        print(f"*** Starting step {i}")
        action, _states = agent.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"* Reward: {reward}, Observation: {observation}, Rewards: {info['rewards']}")
        reward_trajectory.append([ val for key, val in info["rewards"]])
        if reward_cols is None:
            reward_cols = [key for key, val in info["rewards"]]

    reward_trajectory = pd.DataFrame(reward_trajectory, columns = reward_cols)
    reward_trajectory['total_reward'] = reward_trajectory.sum(axis=1)
    plot_trajectory(env.get_output_dict(), reward_trajectory)

if __name__ == "__main__":
    env = SimpleGasStorageEnv(get_example_line)
    obs, info = env.reset()
    print(obs)

    # just a dummy agent alway wanting the maximal inflow
    # just dummy rotating inflows
    inflows = [SimpleGasStorageEnv.MAX_STORAGE_MDOT_KG_PER_S, 0., SimpleGasStorageEnv.MIN_STORAGE_MDOT_KG_PER_S/2]
    fixed_dummy = FixedDummyAgent(inflows)
    run_trajectory(env, fixed_dummy)

    # train our very first agent to maximize the rewards
    trained_agent = train_SB_Agent(env, algorithm=SAC, force_retraining=True)
    run_trajectory(env, trained_agent)
    
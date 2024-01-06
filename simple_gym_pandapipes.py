import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pandapipes as pp 
import pandas as pd
from simple_storage import get_example_line, plot_trajectory
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

class SimpleGasStorageEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["console"]}
    MIN_STORAGE_MDOT_KG_PER_S = -0.05
    MAX_STORAGE_MDOT_KG_PER_S = 0.05
    MAX_TIME_STEPS = 5

    def __init__(self, net_gen_func, mass_storage_id = 0):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # define action and observation space for gym.spaces  .astype(np.float32)  dtype=np.float32
        self.net_gen_func = net_gen_func
        net = net_gen_func() # just for parameter transfer purposes
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
        # - the last action we took
        mass_storage = net.mass_storage.loc[mass_storage_id]
        self.min_m_stored_kg, self.max_m_stored_kg = mass_storage["min_m_stored_kg"],  mass_storage["max_m_stored_kg"],
        self.observation_space = spaces.Box(low=np.array([self.min_m_stored_kg, -1.0, self.min_storage_mdot_kg_per_s]),
                                            high=np.array([self.max_m_stored_kg, 1.0, self.max_storage_mdot_kg_per_s]),
                                            shape=(3,),
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

        # TODO here we'll want something that has an actual learnable pattern
        source_in_flow_df = pd.DataFrame([0.01] * SimpleGasStorageEnv.MAX_TIME_STEPS, columns = ["mdot_kg_per_s"])
        datasource_source_in_flow = DFData(source_in_flow_df)
        const_source = control.ConstControl(self.net, element='source', variable='mdot_kg_per_s',
                                        element_index=self.net.source.index.values,
                                        data_source=datasource_source_in_flow,
                                        profile_name="mdot_kg_per_s")
        
        # defining an OutputWriter to track certain variables
        log_variables = [("mass_storage", "mdot_kg_per_s"), ("res_mass_storage", "mdot_kg_per_s"),
                         ("mass_storage", "m_stored_kg"), ("mass_storage", "filling_level_percent"),
                         ("res_ext_grid", "mdot_kg_per_s"),
                         ("res_source", "mdot_kg_per_s")] 
        
        self.ow = ts.OutputWriter(self.net, log_variables=log_variables, output_path=None)
        self.timeseries_wrapper = TimeseriesWrapper(self.net, self.ow, log_variables)

    def __get_obs(self, init=True):
        mass_storage = self.net.mass_storage.loc[self.mass_storage_id]
        if init:
            mass_stored = mass_storage["init_m_stored_kg"] # be careful, if we have already run a single time step
        else: 
            mass_stored = mass_storage["m_stored_kg"] 

        last_flow = self.net.res_mass_storage.loc[self.mass_storage_id]["mdot_kg_per_s"]
        # 0 for now, since we just have one source
        source_mdot_kg_per_s = self.net.res_source.loc[0]["mdot_kg_per_s"]

        return np.array([mass_stored, last_flow, source_mdot_kg_per_s])

    def __get_info(self):
        return {}
    
    def step(self, action : float):
        # write current action into the data source that the pandapipes simulation is using
        self.storage_flow_df.loc[self.current_timestep] = action
        print(f"* picking action ... {action}")
        # simulating ... 
        self.timeseries_wrapper.run_timestep(self.net, self.current_timestep)
        # evaluating to get the reward ...

        self.current_timestep += 1

        reward = 0
        terminated = False
        observation = self.__get_obs()
        truncated = False
        info = self.__get_info()
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # get all initial parameters set up again
        self.net = self.net_gen_func()
        pp.pipeflow(self.net)
        self.current_timestep = 0
        self.__init_pandapipes_ts()
        return self.__get_obs(), self.__get_info()

    def render(self):
        pass

    def close(self):
        pass

    def get_output_dict(self):
        return self.timeseries_wrapper.output

if __name__ == "__main__":
    env = SimpleGasStorageEnv(get_example_line)
    obs, info = env.reset()
    print(obs)

    # just a dummy agent alway wanting the maximal inflow
    # just dummy rotating inflows
    inflows = [SimpleGasStorageEnv.MAX_STORAGE_MDOT_KG_PER_S, 0., SimpleGasStorageEnv.MIN_STORAGE_MDOT_KG_PER_S/2]
    for i in range(0, 5):
        print(f"*** Starting step {i}")
        env.step(inflows[i % len(inflows)])
    plot_trajectory(env.get_output_dict())
    
import pandapipes as pp
from pandapower.control.basic_controller import Controller
import pandas as pd

import pandapower.timeseries as ts
import pandapower.control as control
from pandapower.timeseries import DFData
import numpy as np 
import os 
# importing a grid from the library
from pandapipes.networks import gas_meshed_square


class StorageController(Controller):
    """
        Example class of a Storage-Controller. Models an abstract mass storage.
    """
    def __init__(self, net, sid, data_source=None, mdot_profile=None, in_service=True,
                 recycle=False, order=0, level=0, duration_timestep_h=1, **kwargs):
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                    initial_pipeflow = True, **kwargs)
        
        # read storage attributes from net
        self.sid = sid  # index of the controlled storage
        self.junction = net.mass_storage.at[sid, "junction"]
        self.mdot_kg_per_s = net.mass_storage.at[sid, "mdot_kg_per_s"]
        self.name = net.mass_storage.at[sid, "name"]
        self.storage_type = net.mass_storage.at[sid, "type"]
        self.in_service = net.mass_storage.at[sid, "in_service"]
        self.scaling = net.mass_storage.at[sid, "scaling"]
        self.applied = False

        # specific attributes
        self.max_m_kg = net.mass_storage.at[sid, "max_m_stored_kg"]
        self.min_m_kg = net.mass_storage.at[sid, "min_m_stored_kg"]
        self.m_stored_kg = net.mass_storage.at[sid, "init_m_stored_kg"]

        # profile attributes
        self.data_source = data_source
        self.mdot_profile = mdot_profile
        self.last_time_step = 0
        self.duration_ts_sec = duration_timestep_h * 3600

    # In a time-series simulation the mass storage should read new flow values from a profile and keep track
    # of its amount of stored mass as depicted below.
    def time_step(self, net, time):
        # keep track of the stored mass (the duration of one time step is given as input to the controller)
        if self.last_time_step is not None:
            # The amount of mass that flowed into or out of the storage in the last timestep is added
            # requested change of mass:
            self.delta_m_kg_req = (self.mdot_kg_per_s * (time - self.last_time_step)
                                   * self.duration_ts_sec)
            # limit by available mass and free capacity in the storage:
            if self.delta_m_kg_req > 0:  # "charging"
                self.delta_m_kg_real = min(self.delta_m_kg_req, self.max_m_kg - self.m_stored_kg)
            else:  # "discharging", delta < 0
                self.delta_m_kg_real = max(self.delta_m_kg_req, self.min_m_kg - self.m_stored_kg)
            self.m_stored_kg += self.delta_m_kg_real
            self.mdot_kg_per_s = self.delta_m_kg_real / ((time - self.last_time_step)
                                                         * self.duration_ts_sec)
        self.last_time_step = time

        # read new values from a profile
        if self.data_source:
            if self.mdot_profile is not None:
                self.mdot_kg_per_s = self.data_source.get_time_step_value(time_step=time,
                                                                          profile_name=self.mdot_profile)
                self.m_stored_kg *= self.scaling * self.in_service
        else: 
            self.mdot_kg_per_s =  -0.05 + np.random.random()*0.1
            
        self.applied = False  # reset applied variable

    # Some convenience methods to calculate indicators for the state of charge:
    def get_stored_mass(self):
        # return the absolute stored mass
        return self.m_stored_kg

    def get_free_stored_mass(self):
        # return the stored mass excl. minimum filling level
        return self.m_stored_kg - self.min_m_kg

    def get_filling_level_percent(self):
        # return the ratio of absolute stored mass and total maximum storable mass in Percent
        return 100 * self.get_stored_mass() / self.max_m_kg

    def get_free_filling_level_percent(self):
        # return the ratio of available stored mass (i.e. excl. min_m_stored_kg) and difference between max and min in Percent
        return 100 * self.get_free_stored_mass() / (self.max_m_kg - self.min_m_kg)

    # Define which values in the net shall be updated
    def write_to_net(self, net):
        # write mdot_kg_per_s, m_stored_kg to the table in the net
        net.mass_storage.at[self.sid, "mdot_kg_per_s"] = self.mdot_kg_per_s
        net.mass_storage.at[self.sid, "m_stored_kg"] = self.m_stored_kg
        net.mass_storage.at[self.sid, "filling_level_percent"] = \
            self.get_free_filling_level_percent()
        # Note: a pipeflow will automatically be conducted in the run_timeseries / run_control procedure.
        # This will then update the result table (net.res_mass_storage).
        # If something was written to net.res_mass_storage in this method here, the pipeflow would overwrite it.

    # In case the controller is not yet converged (i.e. in the first iteration,
    # maybe also more iterations for more complex controllers), the control step is executed.
    # In the example it simply adopts a new value according to the previously calculated target
    # and writes back to the net.
    def control_step(self, net):
        # Call write_to_net and set the applied variable True
        self.write_to_net(net)
        self.applied = True

    # convergence check
    def is_converged(self, net):
        # check if controller already was applied
        return self.applied

def run_simulation(net, ow : ts.OutputWriter, individual_steps=True):
    # starting time series simulation
    np.random.seed(232022 )
    if individual_steps:
        mass_dfs = []
        for i in range(0, 6):
            run_timeseries(net, time_steps=range(i, i+1))
            print("-----------------")
            print(ow.output["mass_storage.m_stored_kg"])
            mass_dfs.append(ow.output["mass_storage.m_stored_kg"])
            print("-----------------")
            print(net.mass_storage.at[0, "m_stored_kg"])
        mass_df = pd.concat(mass_dfs, ignore_index=True)
    else: 
        run_timeseries(net, time_steps=range(0,6))
        mass_df = ow.output["mass_storage.m_stored_kg"]
        ext_grid_flow = ow.output["res_ext_grid.mdot_kg_per_s"]
        source_flow = ow.output["res_source.mdot_kg_per_s"]
        mass_storage_flow = ow.output["mass_storage.mdot_kg_per_s"]
    # run_timeseries(net, time_steps=range(0, 6))
    mass_df.columns = ['mass_storage']
    
    # plotting the state of charge
    import matplotlib.pyplot as plt 
    
    # Assuming the index of your data frames represents time
    time_index = mass_df.index

    # Create a horizontal subplot with 3 subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plotting the data frames on each subplot
    axs[0].plot(time_index, mass_df, color='blue')
    axs[0].set_title('Mass Storage')
    axs[0].set_xlabel('Values')
    axs[0].set_ylabel('Time')

    axs[1].plot(time_index, ext_grid_flow,  color='green', label ="Ext grid flow")
    axs[1].set_xlabel('Values')

    axs[1].plot( time_index, source_flow, color='red', label = "Source flow")
    axs[1].set_xlabel('Values')
    axs[1].legend()

    axs[1].plot( time_index, mass_storage_flow, color='blue', label = "Mass storage flow")
    axs[1].set_title('Flows')
    axs[1].set_xlabel('Values')
    axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":

    from simple_storage import get_example_line
    # loading the network
    net = get_example_line()
    pp.pipeflow(net)

    # creating a simple time series
    # -> that is what I would actually like to see controlled
    framedata = pd.DataFrame([0.01, .05, -0.1, .005, -0.1, 0], columns=['mdot_storage'])
    datasource = ts.DFData(framedata)    # creating storage unit in the grid, which will be controlled by our controller

    # creating an Object of our new build storage controller, controlling the storage unit
    ctrl = StorageController(net=net, sid=0, data_source=datasource, mdot_profile='mdot_storage')

    framedata2 = pd.DataFrame([0.1, 0, 0.1, 0, 0.2, 0.1])
    ds_source = ts.DFData(framedata2)

    target_path_source = os.path.join(os.path.dirname(__file__), 'storage_source_profiles.csv')
    profiles_source = pd.read_csv(target_path_source,
                                            index_col=0)
    ds_source = DFData(profiles_source)

    const_source = control.ConstControl(net, element='source', variable='mdot_kg_per_s',
                                    element_index=net.source.index.values,
                                    data_source=ds_source,
                                    profile_name=net.source.index.values.astype(str))
    
    from pandapipes.timeseries import run_timeseries
    # defining an OutputWriter to track certain variables
    log_variables = [("mass_storage", "mdot_kg_per_s"), ("res_mass_storage", "mdot_kg_per_s"),
                    ("mass_storage", "m_stored_kg"), ("res_ext_grid", "mdot_kg_per_s"),
                    ("res_source", "mdot_kg_per_s")] 
                    #('res_sink', 'mdot_kg_per_s')]
    ow = ts.OutputWriter(net, log_variables=log_variables, output_path=None)

    #run_simulation(net, ow, True)
    run_simulation(net, ow, False)

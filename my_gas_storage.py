import pandapipes as pp
from pandapower.control.basic_controller import Controller
import pandas as pd

import pandapower.timeseries as ts
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
    if individual_steps:
        mass_dfs = []
        for i in range(0, 6):
            run_timeseries(net, time_steps=range(i, i+1))
            print("-----------------")
            print(ow.output["mass_storage.m_stored_kg"])
            mass_dfs.append(ow.output["mass_storage.m_stored_kg"])
            print("-----------------")
        mass_df = pd.concat(mass_dfs, ignore_index=True)
    else: 
        run_timeseries(net, time_steps=range(0,6))
        mass_df = ow.output["mass_storage.m_stored_kg"]
    # run_timeseries(net, time_steps=range(0, 6))
    mass_df.columns = ['mass_storage']
    
    # plotting the state of charge
    import matplotlib.pyplot as plt 
    
    ax = mass_df.plot()
    ax.set_xlabel('Time in 60 min Steps')
    ax.set_ylabel('stored mass in kg')
    ax.legend()
    plt.show()

if __name__ == "__main__":

    # loading the network
    net = gas_meshed_square()
    pp.pipeflow(net)

    # creating a simple time series
    framedata = pd.DataFrame([0.1, .05, -0.1, .005, -0.2, 0], columns=['mdot_storage'])
    datasource = ts.DFData(framedata)

    # creating storage unit in the grid, which will be controlled by our controller
    store_mass = pp.create_mass_storage(net, junction=3,
                                        mdot_kg_per_s=0, init_m_stored_kg=2, min_m_stored_kg=0, max_m_stored_kg=500,
                                        name = "IKIGas_Storage_0",
                                        type="classical mass storage")

    # creating an Object of our new build storage controller, controlling the storage unit
    ctrl = StorageController(net=net, sid=store_mass, data_source=datasource, mdot_profile='mdot_storage')

    from pandapipes.timeseries import run_timeseries
    # defining an OutputWriter to track certain variables
    log_variables = [("mass_storage", "mdot_kg_per_s"), ("res_mass_storage", "mdot_kg_per_s"),
                    ("mass_storage", "m_stored_kg"), ('res_sink', 'mdot_kg_per_s')]
    ow = ts.OutputWriter(net, log_variables=log_variables)

    #run_simulation(net, ow, True)
    run_simulation(net, ow, False)

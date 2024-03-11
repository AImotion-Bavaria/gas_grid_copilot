# prepare just like before
from pandapower import networks as e_nw # electrical networks
net_power = e_nw.example_simple()

import pandapipes as ppipes
import pandapower as ppower
from pandapipes import networks as g_nw # gas networks

from pandapipes.multinet.create_multinet import create_empty_multinet, add_net_to_multinet

net_power = e_nw.example_simple()
net_gas = g_nw.gas_meshed_square()
net_gas.junction.pn_bar = net_gas.ext_grid.p_bar = 30
net_gas.pipe.diameter_m = 0.4
net_gas.controller.rename(columns={'controller': 'object'}, inplace=True) # due to new version
fluid = {'name':'hydrogen', 'cal_value':38.4}
ppipes.create_fluid_from_lib(net_gas, fluid['name'], overwrite=True)
multinet = create_empty_multinet('tutorial_multinet')
add_net_to_multinet(multinet, net_power, 'power_net')
add_net_to_multinet(multinet, net_gas, 'gas_net')

p2g_id_el = ppower.create_load(net_power, bus=3, p_mw=2, name="power to gas consumption")
p2g_id_gas = ppipes.create_source(net_gas, junction=1, mdot_kg_per_s=0, name="power to gas feed in")
g2p_id_gas = ppipes.create_sink(net_gas, junction=1, mdot_kg_per_s=0.1, name="gas to power consumption")
g2p_id_el = ppower.create_sgen(net_power, bus=5, p_mw=0, name="fuel cell feed in")

from pandas import DataFrame
from numpy.random import random
from pandapower.timeseries import DFData

def create_data_source(n_timesteps=10):
    profiles = DataFrame()
    profiles['power to gas consumption'] = random(n_timesteps) * 2 + 1
    profiles['gas to power consumption'] = random(n_timesteps) * 0.1
    ds = DFData(profiles)

    return profiles, ds

profiles, ds = create_data_source(10)

from os.path import join, dirname
from pandapower.timeseries import OutputWriter


#        self.ow = ts.OutputWriter(self.net, log_variables=log_variables, output_path=None)
def create_output_writers(multinet, time_steps=None):
    nets = multinet["nets"]
    ows = dict()
    for key_net in nets.keys():
        ows[key_net] = {}
        if isinstance(nets[key_net], ppower.pandapowerNet):
            log_variables = [('res_bus', 'vm_pu'),
                             ('res_line', 'loading_percent'),
                             ('res_line', 'i_ka'),
                             ('res_bus', 'p_mw'),
                             ('res_bus', 'q_mvar'),
                             ('res_load', 'p_mw'),
                             ('res_load', 'q_mvar')]
            ow = OutputWriter(nets[key_net], time_steps=time_steps,
                              log_variables=log_variables,
                              output_path=None)
            ows[key_net] = ow
        elif isinstance(nets[key_net], ppipes.pandapipesNet):
            log_variables = [('res_sink', 'mdot_kg_per_s'),
                             ('res_source', 'mdot_kg_per_s'),
                             ('res_ext_grid', 'mdot_kg_per_s'),
                             ('res_pipe', 'v_mean_m_per_s'),
                             ('res_junction', 'p_bar'),
                             ('res_junction', 't_k')]
            ow = OutputWriter(nets[key_net], time_steps=time_steps,
                              log_variables=log_variables,
                              output_path=None)
            ows[key_net] = ow
        else:
            raise AttributeError("Could not create an output writer for nets of kind " + str(key_net))
    return ows

ows = create_output_writers(multinet, 10)

from pandapipes.multinet.control.controller.multinet_control import coupled_p2g_const_control, \
    coupled_g2p_const_control
coupled_p2g_const_control(multinet, p2g_id_el, p2g_id_gas,
                          name_power_net="power_net", name_gas_net="gas_net",
                          profile_name='power to gas consumption', data_source=ds,
                          p2g_efficiency=0.7)
coupled_g2p_const_control(multinet, g2p_id_el, g2p_id_gas,
                          name_power_net="power_net", name_gas_net="gas_net",
                          element_type_power="sgen",
                          profile_name='gas to power consumption', data_source=ds,
                          g2p_efficiency=0.65)

print(multinet.controller)
print(net_power.controller)
print(net_gas.controller)

from pandapipes.multinet.timeseries.run_time_series_multinet import run_timeseries
run_timeseries(multinet, time_steps=range(10), output_writers=ows)
pass
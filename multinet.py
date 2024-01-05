from pandapower import networks as e_nw # electrical networks
net_power = e_nw.example_simple()

import pandapipes as pp
from pandapipes import networks as g_nw # gas networks

net_gas = g_nw.gas_meshed_square()
# some adjustments
net_gas.junction.pn_bar = net_gas.ext_grid.p_bar = 30 # 30 bar for _all_ the junctions, I guess
net_gas.pipe.diameter_m = 0.4

# let's get a fluid
pp.create_fluid_from_lib(net_gas, "hydrogen", overwrite=True)

# jetzt kommt der Multinet-Container zum Einsatz - für gekoppelte Netze
from pandapipes.multinet.create_multinet import create_empty_multinet, add_net_to_multinet

multinet = create_empty_multinet("tutorial_multinet")
add_net_to_multinet(multinet, net_power, "power")
add_net_to_multinet(multinet, net_gas, "gas")


print(net_power is multinet.nets['power'])
print(net_gas is multinet.nets['gas'])

# let's build a P2G (power-to-gas) and G2P (gas-to-power) controller
import pandapipes as ppipes 
import pandapower as ppower 

p2g_id_el = ppower.create_load(net_power, bus=3, p_mw = 2, name="power to gas consumption")
p2g_id_gas = ppipes.create_source(net_gas, junction=1, mdot_kg_per_s=0, name="power to gas feed in")

g2p_id_gas = ppipes.create_sink(net_gas, junction=1, mdot_kg_per_s=0.1, name="gas to power consumption")
g2p_id_el = ppower.create_sgen(net_power, bus=5, p_mw=0, name = "fuel cell feed in")

# now for the actual controllers 
from pandapipes.multinet.control.controller.multinet_control import P2GControlMultiEnergy, G2PControlMultiEnergy

p2g_ctrl = P2GControlMultiEnergy(multinet, p2g_id_el, p2g_id_gas, efficiency=0.7, name_power_net="power", name_gas_net="gas")
g2p_ctrl = G2PControlMultiEnergy(multinet, g2p_id_el, g2p_id_gas, efficiency=0.65, name_power_net="power", name_gas_net="gas")

from pandapipes.multinet.control.run_control_multinet import run_control
run_control(multinet)
print(net_gas.source.loc[p2g_id_gas, 'mdot_kg_per_s'])
print(net_power.sgen.loc[g2p_id_el, 'p_mw'])
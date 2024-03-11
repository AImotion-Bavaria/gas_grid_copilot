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

multinet = create_empty_multinet("multi-energy-grid")
add_net_to_multinet(multinet, net_power, "power")
add_net_to_multinet(multinet, net_gas, "gas")


print(net_power is multinet.nets['power'])
print(net_gas is multinet.nets['gas'])

# let's build a P2G (power-to-gas) and G2P (gas-to-power) controller
import pandapipes as ppipes 
import pandapower as ppower 

p2g_id_el = ppower.create_load(net_power, bus=1, p_mw = 2, name="power to gas consumption")
p2g_id_gas = ppipes.create_source(net_gas, junction=2, mdot_kg_per_s=0, name="power to gas feed in")

g2p_id_gas = ppipes.create_sink(net_gas, junction=5, mdot_kg_per_s=0.1, name="gas to power consumption")
g2p_id_el = ppower.create_sgen(net_power, bus=4, p_mw=0, name = "fuel cell feed in")

# now for the actual controllers 
from pandapipes.multinet.control.controller.multinet_control import P2GControlMultiEnergy, G2PControlMultiEnergy

p2g_ctrl = P2GControlMultiEnergy(multinet, p2g_id_el, p2g_id_gas, efficiency=0.7)
g2p_ctrl = G2PControlMultiEnergy(multinet, g2p_id_el, g2p_id_gas, efficiency=0.65)

from pandapipes.multinet.control.run_control_multinet import run_control
run_control(multinet) # startet Lastfluss in pandapower und Rohrfluss in pandapipes
print(net_gas.source.loc[p2g_id_gas, 'mdot_kg_per_s'])
print(net_power.sgen.loc[g2p_id_el, 'p_mw'])



# net_power.res_line.loc[:, ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]].plot.bar()
# net_gas.res_pipe.loc[:, ["v_mean_m_per_s",  "mdot_from_kg_per_s"]].plot.bar() 

import matplotlib.pyplot as plt
def draw_net(net_power, net_gas, ax1, ax2):
    ax1.clear()

    net_power.res_line.loc[:, ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]].plot.bar(ax=ax1, title="Power Grid Lines")
    ax1.set_xticks(net_power.res_line.index, net_power.line.loc[:, "name"])
    ax1.set_ylim([-15, 15])

    ax2.clear()
    net_gas.res_pipe.loc[:, ["v_mean_m_per_s",  "mdot_from_kg_per_s"]].plot.bar(ax=ax2, title = "Gas Grid Pipes")
    ax2.set_xticks(net_gas.res_pipe.index, net_gas.pipe.loc[:, "name"])
    ax2.set_ylim([-0.2, 1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))

#  df.plot.bar(x='Region', rot=0, title='Population', figsize=(15,10), fontsize=12)
draw_net(net_power, net_gas, ax1, ax2)

plt.tight_layout()

# 
from matplotlib.widgets import Button, Slider

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.35)

# Make a horizontal slider to control the frequency.
axmdot = fig.add_axes([0.25, 0.13, 0.65, 0.03])
#axmdot = fig.add_axes([0.25, 0.1, 0.65, 0.03])
mdot_slider = Slider(
    ax=axmdot,
    label='Gas-to-Power: mdot_kg_per_s',
    valmin=0,
    valmax=0.4,
    valinit=0.1
)

# Make a horizontal slider to control the frequency.
# ttuple (left, bottom, width, height)
axpower = fig.add_axes([0.25, 0.08, 0.65, 0.03])
power_slider = Slider(
    ax=axpower,
    label='Power-to-Gas: MW',
    valmin=0,
    valmax=15.0,
    valinit=2.0
)

# The function to be called anytime a slider's value changes
def update(val):
    #line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    # recalc 
    net_power.load.at[p2g_id_el, "p_mw"] = power_slider.val
    net_gas.sink.at[g2p_id_gas, "mdot_kg_per_s"] = mdot_slider.val 
    run_control(multinet)
    draw_net(net_power, net_gas, ax1, ax2)

    # fig.canvas.draw_idle()


# register the update function with each slider
mdot_slider.on_changed(update)
power_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    mdot_slider.reset()
    
button.on_clicked(reset)

plt.show()
pass
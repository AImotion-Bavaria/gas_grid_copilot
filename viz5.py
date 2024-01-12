import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as image
from matplotlib.widgets import Button, Slider

from matplotlib.gridspec import GridSpec
import os 
first_redraw = True

def redraw_net(net, ax):
    ax.clear()
    global first_redraw
    _, axcb = plot_gas_network(net, ax, draw_cb = first_redraw)
    first_redraw = False

def image_plot(net_snapshots):
    fig, ax = plt.subplots(figsize=(14,7))

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, top = 0.85)

    axtime = fig.add_axes([0.25, 0.9, 0.65, 0.03])
    time_slider = Slider(
        ax=axtime,
        label='Time steps',
        valmin=0,
        valmax=len(net_snapshots),
        valinit=5,
        valfmt='%0.0f',
        valstep=range(0,len(net_snapshots))
    )

    time_slider.on_changed(lambda i : redraw_net(net_snapshots[i], ax))
    time_slider.set_val(0)
    plt.show()

import pickle
from simple_storage import get_example_line
import matplotlib as mpl
import matplotlib.style as mplstyle

from viz_step_by_step import plot_gas_network 
import pandapipes as pp

if __name__ == "__main__":   
    # start out with our net 
    net = get_example_line()
    mplstyle.use('fast')
    mpl.rcParams['path.simplify_threshold'] = 1.0
    mpl.rcParams['path.simplify'] = True

    pp.pipeflow(net)

    net_jsons = []
    net_snapshots = []
    for i in range(0, 10): 
        net.mass_storage.loc[0]["m_stored_kg"] += 10
        net.source.loc[0]["mdot_kg_per_s"] += 0.05
        pp.pipeflow(net)
        net_jsons.append(pp.to_json(net, None))
        net_snapshots.append(pp.from_json_string(net_jsons[-1]))
    image_plot(net_snapshots)
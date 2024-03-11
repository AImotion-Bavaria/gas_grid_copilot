import matplotlib.pyplot as plt
# prepare just like before
from pandapower import networks as e_nw # electrical networks
import pandapipes as ppipes
from pandapipes import plotting as plot
from pandapower import plotting as pow_plot
import pandapower as ppower
from pandapipes import networks as g_nw # gas networks
import numpy as np
from pandapipes.multinet.create_multinet import create_empty_multinet, add_net_to_multinet


def plot_gas_network(net, ax, draw_cb=True):
    # plot.simple_plot(line, plot_sinks=True, plot_sources=True, junction_color="blue", pipe_color="black")
    # do it step instead 

    # create a list of simple collections
    simple_collections = plot.create_simple_collections(net, as_dict=True, 
                                                        plot_sinks=True, 
                                                        plot_sources=True,
                                                        pipe_width=2.0, 
                                                        pipe_color="black",
                                                        junction_color="silver",
                                                        sink_size=1.0,
                                                        source_size=1.0
                                                        )

    # convert dict to values
    pipe_collection = simple_collections["pipe"]
    pipe_collection.set_colors("orange")
    #pipe_collection.set_array(net.res_pipe["mdot_to_kg_per_s"])
    #pipe_collection.set_linewidths(5.)
    simple_collections = list(simple_collections.values())

    plot.draw_collections(simple_collections, ax = ax)

    if draw_cb:
        axcb = plt.colorbar(pipe_collection, ax = ax, boundaries = np.linspace(-0.1,0.1,1000))
    else:
        axcb = None
    return ax, axcb

def plot_power_network(net, ax):
    # create a list of simple collections
    lc = pow_plot.create_line_collection(net, net.line.index, color="blue", zorder=1) #create lines
    bc = pow_plot.create_bus_collection(net, net.bus.index, size=80, color="blue", zorder=2) #create buses
    
    pow_plot.draw_collections([lc, bc], ax = ax) # plot lines and buses

def insert_geo_data_gas(net_gas):
    net_gas.junction_geodata.iloc[0,:] = [0,1]
    net_gas.junction_geodata.iloc[1,:] = [3,1]
    net_gas.junction_geodata.iloc[2,:] = [1,1]
    net_gas.junction_geodata.iloc[3,:] = [1,2]
    net_gas.junction_geodata.iloc[4,:] = [2,2]
    net_gas.junction_geodata.iloc[5,:] = [2,1]


if __name__ == "__main__":
    net_power = e_nw.example_simple()
    net_gas = g_nw.gas_meshed_square()
    insert_geo_data_gas(net_gas)

    ppipes.pipeflow(net_gas)
    ppower.runpp(net_power)
    fig=plt.figure(figsize=(15,11))
    ax = fig.add_subplot()

    # pow_plot.simple_plot(net_power, ax=ax, 
    #                      plot_loads=True,
    #                      plot_line_switches=True, 
    #                      plot_sgens=True, 
    #                      plot_gens=True, line_color = "blue")
    #plot_power_network(net_power, ax)
    #ax2 = fig.add_subplot()
    plot_gas_network(net_gas, ax, draw_cb=False)
   
    plt.show()
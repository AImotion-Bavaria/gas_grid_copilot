import pandapipes as pp
# import the plotting module
import ikigas_plots as plot
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['image.cmap']='coolwarm'
import numpy as np

import simple_storage
# write the mass_flow as a label 
# maybe the length below
def write_pipe_labels(net : pp.pandapipesNet):
    for i in net.pipe.index:
        geodata = net.pipe_geodata.loc[i].coords
        x, y = center_gravity = np.mean(geodata, axis=0)
        mass_flow = net.res_pipe.loc[i]["mdot_to_kg_per_s"]
        plt.text(x, y+.05, "$ \dot{m} = $"+ f"{np.round(mass_flow, 2)} kg/s", fontsize=15, horizontalalignment='center')

def write_inflow_labels(net : pp.pandapipesNet):
    for i in net.source.index:
        geodata = net.junction_geodata.loc[net.source.loc[0].junction]
        x, y = geodata
        y -= 0.25
        mass_flow = net.res_source.loc[i]["mdot_kg_per_s"]
        plt.text(x, y, "Inflow:\n $\dot{m} = $"+ f"{np.round(mass_flow, 2)} kg/s", fontsize=15, horizontalalignment='center')

def write_outflow_labels(net : pp.pandapipesNet):
    for i in net.sink.index:
        geodata = net.junction_geodata.loc[net.sink.loc[0].junction]
        x, y = geodata
        y += 0.15
        mass_flow = net.res_sink.loc[i]["mdot_kg_per_s"]
        plt.text(x, y, "Outflow:\n $\dot{m} = $"+ f"{np.round(mass_flow, 2)} kg/s", fontsize=15, horizontalalignment='center')


from ikigas_plots import create_mass_storage_collection
def get_mass_storage_collections(net, respect_in_service = False):
    if len(net.mass_storage) > 0:
        idx = net.mass_storage[net.mass_storage.in_service].index if respect_in_service else net.mass_storage.index
        storage_colls = create_mass_storage_collection(net, mass_storages=idx, size=0.08,
                                                patch_edgecolor='black', line_color='black',
                                                linewidths=2.0, orientation=0)
    return storage_colls

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
                                                        sink_size=2.0,
                                                        source_size=2.0
                                                        )

    # convert dict to values
    pipe_collection = simple_collections["pipe"]
    pipe_collection.set_colors(None)
    pipe_collection.set_array(net.res_pipe["mdot_to_kg_per_s"])
    pipe_collection.set_linewidths(5.)
    simple_collections = list(simple_collections.values())

    # add additional collections to the list
    junction_mass_storage_collection = plot.create_junction_collection(net, junctions=[2], 
                                                                    patch_type="rect",
                                                                    size=0.05, color="green", 
                                                                    zorder=200)
    mass_storage_collection = get_mass_storage_collections(net)

    #source_colls = create_source_collection(net, sources=idx, size=source_size,
    #                                                patch_edgecolor='black', line_color='black',
    #                                                linewidths=pipe_width, orientation=0)
    simple_collections.append(mass_storage_collection)
    simple_collections.append([junction_mass_storage_collection])

    plot.draw_collections(simple_collections, ax = ax)
    write_pipe_labels(net)
    write_inflow_labels(net)
    write_outflow_labels(net)
    if draw_cb:
        axcb = plt.colorbar(pipe_collection, ax = ax, boundaries = np.linspace(-0.1,0.1,1000))
    else:
        axcb = None
    return ax, axcb

from PIL import Image
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def get_network_image(net):
    old_backend = matplotlib.rcParams['backend'] 
    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(16,7))
    plot_gas_network(net, ax)
    img = fig2img(fig)
    matplotlib.use(old_backend)
    return img

if __name__ == "__main__":
    line = simple_storage.get_example_line()
    pp.pipeflow(line)
    # plot list of all collections
    fig, ax = plt.subplots(figsize=(16,7))
    plot_gas_network(line, ax)
   
    plt.show()
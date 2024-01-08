import pandapipes as pp

# create an empty network
net = pp.create_empty_network(fluid="lgas")

# create network elements, such as junctions, external grid, pipes, valves, sinks and sources
junction1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Connection to External Grid", geodata=(0, 0))
junction2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 2", geodata=(2, 0))
junction3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 3", geodata=(7, 4))
junction4 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 4", geodata=(7, -4))
junction5 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 5", geodata=(5, 3))
junction6 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 6", geodata=(5, -3))

ext_grid = pp.create_ext_grid(net, junction=junction1, p_bar=1.1, t_k=293.15, name="Grid Connection")

pipe1 = pp.create_pipe_from_parameters(net, from_junction=junction1, to_junction=junction2, length_km=10, diameter_m=0.3, name="Pipe 1", geodata=[(0, 0), (2, 0)])
pipe2 = pp.create_pipe_from_parameters(net, from_junction=junction2, to_junction=junction3, length_km=2, diameter_m=0.3, name="Pipe 2", geodata=[(2, 0), (2, 4), (7, 4)])
pipe3 = pp.create_pipe_from_parameters(net, from_junction=junction2, to_junction=junction4, length_km=2.5, diameter_m=0.3, name="Pipe 3", geodata=[(2, 0), (2, -4), (7, -4)])
pipe4 = pp.create_pipe_from_parameters(net, from_junction=junction3, to_junction=junction5, length_km=1, diameter_m=0.3, name="Pipe 4", geodata=[(7, 4), (7, 3), (5, 3)])
pipe5 = pp.create_pipe_from_parameters(net, from_junction=junction4, to_junction=junction6, length_km=1, diameter_m=0.3, name="Pipe 5", geodata=[(7, -4), (7, -3), (5, -3)])

valve = pp.create_valve(net, from_junction=junction5, to_junction=junction6, diameter_m=0.05, opened=True)

sink = pp.create_sink(net, junction=junction4, mdot_kg_per_s=0.545, name="Sink 1")

source = pp.create_source(net, junction=junction3, mdot_kg_per_s=0.234)
pp.pipeflow(net)

# import the plotting module
import pandapipes.plotting as plot
import matplotlib.pyplot as plt

# create additional junction collections for junctions with sink connections and junctions with valve connections
junction_sink_collection = plot.create_junction_collection(net, junctions=[3], patch_type="circle", size=0.1, color="orange", zorder=200)
junction_source_collection = plot.create_junction_collection(net, junctions=[2], patch_type="circle", size=0.1, color="green", zorder=200)
junction_valve_collection = plot.create_junction_collection(net, junctions=[4, 5], patch_type="rect",size=0.1, color="red", zorder=200)

# create additional pipe collection
pipe_collection = plot.create_pipe_collection(net, pipes=[3,4], linewidths=3., zorder=100)
# plot network
# plot.simple_plot(net, plot_sinks=True, plot_sources=True, sink_size=4.0, source_size=4.0)
# # plot collections of junctions and pipes
# plot.draw_collections([junction_sink_collection, junction_source_collection, junction_valve_collection, pipe_collection],  figsize=(8,6))
# plt.show()

# # create a list of simple collections
simple_collections = plot.create_simple_collections(net, as_dict=False)

# # add additional collections to the list
simple_collections.append([junction_sink_collection, junction_source_collection, junction_valve_collection, pipe_collection])

# # plot list of all collections
plot.draw_collections(simple_collections)
plt.show()

import simple_storage
line = simple_storage.get_example_line()
pp.pipeflow(line)
# plot.simple_plot(line, plot_sinks=True, plot_sources=True, junction_color="blue", pipe_color="black")
# do it step instead 
 
# create a list of simple collections
simple_collections = plot.create_simple_collections(line, as_dict=False, plot_sinks=True, plot_sources=True)

# add additional collections to the list
junction_mass_storage_collection = plot.create_junction_collection(line, junctions=[2], patch_type="rect",size=0.05, color="green", zorder=200)

simple_collections.append([junction_mass_storage_collection])

# plot list of all collections
plot.draw_collections(simple_collections)
plt.show()
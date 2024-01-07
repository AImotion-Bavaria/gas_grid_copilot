import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pandapipes as pp
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.colors as mcolors
from simple_storage import get_example_line

def visualize_grid(net):
    #G = nx.MultiGraph()
    G = pgv.AGraph()
    
    # Add junctions
    for junction in net.junction.index:
        G.add_node(junction, color='white', shape='rect', style='filled', label=f'Junction {junction}')

    # Add external grid
    ext_grid_node = list(net.ext_grid.index)[0]
    G.add_node(ext_grid_node, color='black', shape='rect', label='External Grid')

    # Add source
    source_node = list(net.source.index)[0]
    G.add_node(source_node, color='white', shape='circle', label='Source')

    # Add mass storage
    storage_node = list(net.mass_storage.index)[0]
    fill_percentage = 0.5 # net.mass_storage.at[storage_node, 'p_mw'] / net.mass_storage.at[storage_node, 'max_p_mw']
    storage_color = mcolors.to_rgba(f'C{int(fill_percentage * 10)}')
    G.add_node(storage_node, color=storage_color, shape='rect', label=f'Mass Storage\n({round(fill_percentage * 100)}%)')

    # Add pipes with thickness based on mass flow
    for pipe in net.pipe.index:
        from_junction = net.pipe.at[pipe, 'from_junction']
        to_junction = net.pipe.at[pipe, 'to_junction']
        mass_flow = net.res_pipe.at[pipe, 'mdot_from_kg_per_s']
        G.add_edge(from_junction, to_junction, weight=mass_flow, label=f'{round(mass_flow, 2)} kg/s')

    # G.layout(prog="dot")  # use dot
    G.layout(prog="dot")  # use dot
    G.draw("file.png")
    
    # Set up layout and plot
    #pos = graphviz_layout(G, prog="dot")
    #edge_labels = nx.get_edge_attributes(G, 'label')
    #node_labels = nx.get_node_attributes(G, 'label')
    #nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=5000, node_color='lightgray', font_size=8)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')


    #plt.show()

# Create the network using get_example_line
net = get_example_line()
pp.pipeflow(net)
visualize_grid(net)

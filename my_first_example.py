import pandapipes as pp

def get_example_fork():
    # let's create the net first
    net = pp.create_empty_network(fluid="hydrogen")
    pn_bar = 30
    norm_temp = 293.15 # in K, would be 20° C

    #create junctions

    j = dict()
    for i in range(0, 6):
        j[i] = pp.create_junction(net, pn_bar=pn_bar, tfluid_k=norm_temp, name=f"Junction {i}")


    #create junction elements
    # external grid access has a little higher pressure 
    ext_grid = pp.create_ext_grid(net, junction=j[0], p_bar=30.1, t_k=293.15, name="Grid Connection 1")
    ext_grid2 = pp.create_ext_grid(net, junction=j[1], p_bar=30.1, t_k=293.15, name="Grid Connection 2")
    sink = pp.create_sink(net, junction=j[5], mdot_kg_per_s=0.1, name="Sink")

    # now for the actual pipes?
    #create branch elements
    pp.create_pipe_from_parameters(net, from_junction=j[0], to_junction=j[2], length_km = 10, diameter_m=0.4, name="Pipe 0")
    pp.create_pipe_from_parameters(net, from_junction=j[1], to_junction=j[3], length_km = 10, diameter_m=0.4, name="Pipe 1")
    pp.create_pipe_from_parameters(net, from_junction=j[2], to_junction=j[4], length_km = 100, diameter_m=0.4, name="Pipe 2")
    pp.create_pipe_from_parameters(net, from_junction=j[3], to_junction=j[4], length_km = 100, diameter_m=0.4, name="Pipe 3")
    pp.create_pipe_from_parameters(net, from_junction=j[4], to_junction=j[5], length_km = 5, diameter_m=0.4, name="Pipe 4")

    #valve1 = pp.create_valve(net, from_junction=j[1], to_junction=j[3], diameter_m=0.4, opened=True, name="Valve")
    #valve2 = pp.create_valve(net, from_junction=j[2], to_junction=j[4], diameter_m=0.4, opened=True, name="Valve")

    pp.pipeflow(net)

    # now for some of the results:
    return net

if __name__ == "__main__":
    net = get_example_fork()
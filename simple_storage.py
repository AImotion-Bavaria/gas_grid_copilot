import pandapipes as pp

def get_example_line():
    # let's create the net first
    net = pp.create_empty_network(fluid="hydrogen")
    pn_bar = 30
    norm_temp = 293.15 # in K, would be 20° C

    #create junctions

    j = dict()
    for i in range(0, 3):
        j[i] = pp.create_junction(net, pn_bar=pn_bar, tfluid_k=norm_temp, name=f"Junction {i}")


    #create junction elements
    ext_grid = pp.create_ext_grid(net, junction=j[0], p_bar=pn_bar, t_k=293.15, name="Grid Connection 1")
    source = pp.create_source(net, junction=j[1], mdot_kg_per_s=0.2, name="My source")
    pp.create_mass_storage(net, junction=j[2], mdot_kg_per_s=0.1, 
                           init_m_stored_kg=2, 
                           min_m_stored_kg=0, max_m_stored_kg=500,
                           name = "Test Storage",
                           type="Classical mass storage")

    # now for the actual pipes?
    #create branch elements
    pp.create_pipe_from_parameters(net, from_junction=j[0], to_junction=j[1], length_km = 10, diameter_m=0.4, name="Pipe 0")
    pp.create_pipe_from_parameters(net, from_junction=j[1], to_junction=j[2], length_km = 10, diameter_m=0.4, name="Pipe 1")

    #valve1 = pp.create_valve(net, from_junction=j[1], to_junction=j[3], diameter_m=0.4, opened=True, name="Valve")
    #valve2 = pp.create_valve(net, from_junction=j[2], to_junction=j[4], diameter_m=0.4, opened=True, name="Valve")

    pp.pipeflow(net)

    # now for some of the results:
    return net

if __name__ == "__main__":
    net = get_example_line()
import pandapipes as pp
import pandas as pd
import numpy as np

# plotting the state of charge
import matplotlib.pyplot as plt 

def get_minimal_line():
    # let's create the net first
    net = pp.create_empty_network(fluid="hydrogen")
    pn_bar = 30
    norm_temp = 293.15 # in K, would be 20° C

    #create junctions
    j = dict()
    for i in range(0, 4):
        j[i] = pp.create_junction(net, pn_bar=pn_bar, tfluid_k=norm_temp, name=f"Junction {i}")
        
    #grid und source hängen an junction 0, damit wir Masseerhaltung sehen
    ext_grid = pp.create_ext_grid(net, junction=j[0], p_bar=pn_bar, t_k=293.15, name="Grid Connection 1")
    source = pp.create_source(net, junction=j[1], mdot_kg_per_s=0.15, name="My source")
    sink = pp.create_sink(net, junction=j[3], mdot_kg_per_s=1.5, name="My sink")

    pp.create_pipe_from_parameters(net, from_junction=j[0], to_junction=j[2], length_km = 1, diameter_m=0.4, name="Pipe 0")
    pp.create_pipe_from_parameters(net, from_junction=j[1], to_junction=j[2], length_km = 1, diameter_m=0.4, name="Pipe 1")    
    pp.create_pipe_from_parameters(net, from_junction=j[2], to_junction=j[3], length_km = 100, diameter_m=0.4, name="Pipe 2")

    pp.pipeflow(net)

    # now for some of the results:
    return net

if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=200)
    pd.set_option('display.expand_frame_repr', False)
    net = get_minimal_line()

    # inspect the first pipe as a sanity check
    from pandapipes.idx_branch import * 
    from pandapipes.idx_node import * 
    from pandapipes.constants import NORMAL_PRESSURE, NORMAL_TEMPERATURE

    # looking for the formula for the pressure loss 
    node_pit, branch_pit = net["_pit"]["node"], net["_pit"]["branch"]
    lam = branch_pit[0, LAMBDA]
    rho = net.fluid.get_density(293.15).item()

    v_mps = branch_pit[0, VINIT] 
    v_gas_from = net.res_pipe.loc[0, "v_from_m_per_s"]
    p_from_bar = net.res_pipe.loc[0, "p_from_bar"]
    t = net.res_pipe.loc[0, "t_from_k"]
    d = net.pipe.loc[0, "diameter_m"]
    dl = net.pipe.loc[0, "length_km"] # *1000
    A = (d/2)**2 * np.pi

    # first, is the mass flow equal to v_N / (A * rho_N)?
    rho_N = net.fluid.get_density(NORMAL_TEMPERATURE).item()
    print("Mass flow: ", net.res_pipe.loc[0, "mdot_from_kg_per_s"])
    print("From velocity: ", v_mps * (A * rho))  

    # now for the pressure loss in the first pipe 
    dp = - lam * (rho * v_gas_from**2) * (NORMAL_PRESSURE/p_from_bar) * (t / NORMAL_TEMPERATURE)/ (2*d)
    # pressure difference applied to pressure at junction 0 should match pressure at junction 2
    print(f"Starting from pressure {p_from_bar} and applying a difference of {dp}")
    print(p_from_bar + dp)
    print("Is this equal to the pressure at junction 2? ")
    print(net.res_pipe.loc[0, "p_to_bar"])

    # Attempt 2 with other formula
    dp = - (lam * (rho * v_gas_from**2) * dl ) / (2*d)
    print(f"What about this? ")
    print(p_from_bar + dp)
    pass
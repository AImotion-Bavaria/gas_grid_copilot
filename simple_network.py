import pandapipes as pp
#create empty net
net = pp.create_empty_network(fluid="lgas")
pn_bar = 1.05
norm_temp = 293.15 # in K, would be 20° C

#create junctions
j1 = pp.create_junction(net, pn_bar=pn_bar, tfluid_k=norm_temp, name="Junction 1")
j2 = pp.create_junction(net, pn_bar=pn_bar, tfluid_k=norm_temp, name="Junction 2")
j3 = pp.create_junction(net, pn_bar=pn_bar, tfluid_k=norm_temp, name="Junction 3")

#create junction elements
# external grid access has a little higher pressure 
ext_grid = pp.create_ext_grid(net, junction=j1, p_bar=1.1, t_k=293.15, name="Grid Connection")
sink = pp.create_sink(net, junction=j3, mdot_kg_per_s=0.045, name="Sink")

# now for the actual pipes?
#create branch elements
pipe = pp.create_pipe_from_parameters(net, from_junction=j1, to_junction=j2, length_km=0.1, diameter_m=0.05, name="Pipe")
valve = pp.create_valve(net, from_junction=j2, to_junction=j3, diameter_m=0.05, opened=True, name="Valve")

pp.pipeflow(net)

# now for some of the results:
print(net)
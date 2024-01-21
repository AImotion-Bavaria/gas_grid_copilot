# now create an actual dzn for this neural network 
# let's start really simple and use global variables for it 
def convert_to_minizinc_data(model, suffix=None):
    input_neurons = model.fc1.weight.shape[1]
    h1_neurons = model.fc1.weight.shape[0]
    h2_neurons = model.fc2.weight.shape[0]
    output_neurons = model.fc3.weight.shape[0]
    
    total_neurons = input_neurons + h1_neurons + h2_neurons + output_neurons
    total_edges = sum([fc.weight.shape[0] * fc.weight.shape[1] for fc in [model.fc1, model.fc2, model.fc3]])
    # h1 = model.relu((model.fc1.weight @ scaled_in) + model.fc1.bias.reshape(-1, 1))
    # h2 = model.relu((model.fc2.weight @ h1) + model.fc2.bias.reshape(-1, 1))
    # out = model.relu((model.fc3.weight @ h2) + model.fc3.bias.reshape(-1, 1))
    
    # input ids are relatively trivial
    input_ids = [input_id + 1 for input_id in range(0, input_neurons)]
    output_ids = [input_id + 1 for input_id in range(total_neurons-output_neurons, total_neurons)]

    # let's start with biases, all hidden neurons get a bias, input and output are zero
    bias = [0] * total_neurons
    current_node_id = 1
    # at the same time let's look at the edges
    edge_weight = [0] * total_edges
    edge_parent = [0] * total_edges
    first_edge = [1] * total_neurons

    current_edge_id = 0
    node_offsets = [0] * 4 # for each layer, including the input
    # 0 in the first position is okay for inputs
    for index, fc in enumerate([model.fc1, model.fc2, model.fc3]):
        layer_index = index + 1 # we start in layer 1 
        node_offsets[layer_index] = current_node_id
        for node_index, bias_val in enumerate(fc.bias):
            bias[current_node_id] = bias_val.detach().numpy().item()
            # that also gives me the next node in this respective layer!
            first_edge[current_node_id] = current_edge_id + 1
            for ingoing in range(fc.weight.shape[1]):
                edge_weight[current_edge_id] = fc.weight[node_index, ingoing].detach().numpy().item()
                # this has to be the node ID, not the ingoing index between 0 and 40
                edge_parent[current_edge_id] = node_offsets[layer_index-1] + ingoing + 1
                current_edge_id += 1 
            current_node_id += 1
            #node_offsets[layer_index] = 
    if suffix:
        suffix = "_"+suffix
    else:
        suffix = ""

    output_dict = {f"N_Neurons{suffix}" : total_neurons,
                   f"N_Edges{suffix}" : total_edges,
                   f"input_ids{suffix}" : input_ids,
                   f"output_ids{suffix}" : output_ids,
                   f"bias{suffix}" : bias,
                   f"edge_weight{suffix}" : edge_weight,
                   f"edge_parent{suffix}" : edge_parent,
                   f"first_edge{suffix}" : first_edge}
    
    return output_dict

def get_minizinc_nnet_parameters(suffix : str): 
    """for every NN in MZN, we need all parameters in the main model
    therefore, a convenience method is useful

    Args:
        prefix (str): an identifier for the NN (e.g., time_to_demand)
    """
    parameters = []
    parameters.append(f"int: N_Neurons_{suffix};")
    parameters.append(f"set of int: Neurons_{suffix} = 1..N_Neurons_{suffix};")
    parameters.append(f"int: N_Edges_{suffix};")
    parameters.append(f"set of int: Edges_{suffix} = 1..N_Edges_{suffix};")
    parameters.append(f"array[int] of int: input_ids_{suffix};")
    parameters.append(f"array[int] of int: output_ids_{suffix};")
    parameters.append(f"array[Neurons_{suffix}] of float: bias_{suffix};")
    parameters.append(f"array[Neurons_{suffix}] of Edges_{suffix}: first_edge_{suffix};")
    parameters.append(f"array[Edges_{suffix}] of float: edge_weight_{suffix};")
    parameters.append(f"array[Edges_{suffix}] of Neurons_{suffix}: edge_parent_{suffix};")
    return parameters, "\n".join(parameters)
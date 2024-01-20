import torch as th
import os
import pickle  
from simple_nn import SimpleNN

model = SimpleNN().cpu()
model.load_state_dict(th.load("trained_model.pth"))
model.eval()
# load scaler as well 
scaler_data_name = "scaler.pl"
scaler_data_name = os.path.join(os.path.dirname(__file__), scaler_data_name)

with open(scaler_data_name, 'rb') as file:
    scaler = pickle.load(file)

# trying to reproduce the scaler 
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
x_min = 0.
x_max = 9.    

# retrieved from MiniZinc
predicted_flow = { 0: 0.2772106521173153, 1: 1.379844442215824, 2: 1.999999970169527, 3: 0.9006037195297723, 4: 0.7141275408106543, 5: 1.859374719504125, 6: 1.64020968488827, 7: 0.4218001980599828, 8: 0.01612432673573494, 9: 0.01612432673573494}
for i in range(0, 10): 
    y_i = model(th.tensor(scaler.transform([[i]]), dtype=th.float32)).detach().numpy().item()
    print(f"Time step {i}: Normalized {scaler.transform([[i]]).item()} Output: {y_i}")
    x_std = ( i - x_min) / (x_max - x_min)
    x_scaled = x_std * (x_max - x_min) + x_min
    print("Reproduced scale: ", x_std)
    print("output from minizinc ", predicted_flow[i])
print("Hi")

# now create an actual dzn for this neural network 
# let's start really simple and use global variables for it 
def convert_to_dzn(model):
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
    biases = [0] * total_neurons
    current_node_id = 1
    # at the same time let's look at the edges
    edge_weights = [0] * total_edges
    edge_parent = [0] * total_edges
    first_edge = [1] * total_neurons

    current_edge_id = 0
    node_offsets = [0] * 4 # for each layer, including the input
    # 0 in the first position is okay for inputs
    for index, fc in enumerate([model.fc1, model.fc2, model.fc3]):
        layer_index = index + 1 # we start in layer 1 
        node_offsets[layer_index] = current_node_id
        for node_index, bias_val in enumerate(fc.bias):
            biases[current_node_id] = bias_val.detach().numpy().item()
            # that also gives me the next node in this respective layer!
            first_edge[current_node_id] = current_edge_id + 1
            for ingoing in range(fc.weight.shape[1]):
                edge_weights[current_edge_id] = fc.weight[node_index, ingoing].detach().numpy().item()
                # this has to be the node ID, not the ingoing index between 0 and 40
                edge_parent[current_edge_id] = node_offsets[layer_index-1] + ingoing + 1
                current_edge_id += 1 
            current_node_id += 1
            #node_offsets[layer_index] = 
    return input_ids, output_ids, biases, edge_weights, edge_parent, first_edge

input_ids, output_ids, biases, edge_weights, edge_parent, first_edge = convert_to_dzn(model)

# dummy DZN export for now - if we have the encoding right, the rest is fancy-pants work
print(f"input_ids = {input_ids};")
print(f"output_ids = {output_ids};")
print(f"biases = {biases};")
print(f"edge_weights = {edge_weights};")
print(f"edge_parent = {edge_parent};")
print(f"first_edge = {first_edge};")
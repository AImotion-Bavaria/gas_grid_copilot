set of int: Time = 0..9;
float: min_T = min(Time);
float: max_T = max(Time); 

function var float: normalize(var float: value, float: x_min, float: x_max) =
(value - x_min) / (x_max - x_min);

array[Time] of var float: scaled_Time;
array[Time] of var float: predicted_flow;

% x_std = ( i - x_min) / (x_max - x_min)
constraint forall(t in Time) (scaled_Time[t] = normalize(t, min_T, max_T));

constraint forall(t in Time) (
            predicted_flow[t] = 
                neural_net_1d(scaled_Time[t], input_ids, output_ids, biases, edge_weights, edge_parent, first_edge, NT_RELU)
           );

%var float: output_prediction;
%constraint output_prediction = neural_net_1d(scaled_Time[1], input_ids, output_ids, biases, edge_weights, edge_parent, first_edge, NT_RELU, all_nodes);
% =================================
% Neural network parameter definitions that stay the same for all calls to the function!
int: N_Neurons = 82;
set of int: Neurons = 1..N_Neurons;
int: N_Edges = 1680;
set of int: Edges = 1..N_Edges;

array[int] of int: input_ids; % = [1];
array[int] of int: output_ids; % = [N_Neurons];

array[Neurons] of float: biases;
array[Neurons] of Edges: first_edge;
%array[Neurons] of var float: all_nodes; % for debugging

array[Edges] of float: edge_weights;
array[Edges] of Neurons: edge_parent;

% =================================
include "neural_net.mzn";

function var float: neural_net_1d(var float: input,
                   array[int] of int: input_ids,		         
		               array[int] of int: output_ids,
		               array[int] of float: bias,
		               array[int] of float: edge_weight,
		               array[int] of int: edge_parent,
		               array[int] of int: first_edge,
	                 NEURON_TYPE: neuron_type ) =
%	                 array[int] of var float: all_nodes) =		
let {
  array[index_set(input_ids)] of var float: inputs;
  array[index_set(output_ids)] of var float: outputs;
  constraint inputs[1] == input;
  constraint neural_net(inputs, input_ids, outputs, output_ids, bias, edge_weight, edge_parent, first_edge, NT_RELU)
  %constraint neural_net2(inputs, input_ids, outputs, output_ids, bias, edge_weight, edge_parent, first_edge, NT_RELU, all_nodes);
}
in 
  outputs[1];
set of int: Time = 0..9;
float: min_T = min(Time);
float: max_T = max(Time); 

function var float: normalize(var float: value, float: x_min, float: x_max) =
(value - x_min) / (x_max - x_min);

array[Time] of var float: scaled_Time;
array[Time] of var float: predicted_sink_flow;
array[Time] of var float: predicted_source_flow;

% x_std = ( i - x_min) / (x_max - x_min)
constraint forall(t in Time) (scaled_Time[t] = normalize(t, min_T, max_T));

constraint forall(t in Time) (
              predicted_sink_flow[t] = time_to_sink_prediction(scaled_Time[t]) /\
              predicted_source_flow[t] = time_to_source_prediction(scaled_Time[t])
           );


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
  
%% NNet 1: Predict source from time

% Helper for better readability:

function var float: time_to_source_prediction(var float: input) = 
  neural_net_1d(input, input_ids_source_net, output_ids_source_net, bias_source_net, edge_weight_source_net, edge_parent_source_net, first_edge_source_net, NT_RELU);
  
int: N_Neurons_source_net;
set of int: Neurons_source_net = 1..N_Neurons_source_net;
int: N_Edges_source_net;
set of int: Edges_source_net = 1..N_Edges_source_net;
array[int] of int: input_ids_source_net;
array[int] of int: output_ids_source_net;
array[Neurons_source_net] of float: bias_source_net;
array[Neurons_source_net] of Edges_source_net: first_edge_source_net;
array[Edges_source_net] of float: edge_weight_source_net;
array[Edges_source_net] of Neurons_source_net: edge_parent_source_net;

%% NNet 2: Predict sink from time
function var float: time_to_sink_prediction(var float: input) = 
  neural_net_1d(input, input_ids_sink_net, output_ids_sink_net, bias_sink_net, edge_weight_sink_net, edge_parent_sink_net, first_edge_sink_net, NT_RELU);
  
int: N_Neurons_sink_net;
set of int: Neurons_sink_net = 1..N_Neurons_sink_net;
int: N_Edges_sink_net;
set of int: Edges_sink_net = 1..N_Edges_sink_net;
array[int] of int: input_ids_sink_net;
array[int] of int: output_ids_sink_net;
array[Neurons_sink_net] of float: bias_sink_net;
array[Neurons_sink_net] of Edges_sink_net: first_edge_sink_net;
array[Edges_sink_net] of float: edge_weight_sink_net;
array[Edges_sink_net] of Neurons_sink_net: edge_parent_sink_net;


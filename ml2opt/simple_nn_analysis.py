import torch as th
import os
import pickle  
from simple_nn import SimpleNN

from pandapipes.ml2opt.torch2minizinc import convert_to_minizinc_data, get_minizinc_nnet_parameters

def load_model(model_path):
    model = SimpleNN().cpu()
    
    import os
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    model.load_state_dict(th.load(model_path, map_location=th.device('cpu')))
    model.eval()
    return model 

model_source = load_model("trained_model_source.pth")
model_sink = load_model("trained_model_sink.pth")

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
    y_i = model_source(th.tensor(scaler.transform([[i]]), dtype=th.float32)).detach().numpy().item()
    print(f"Time step {i}: Normalized {scaler.transform([[i]]).item()} Output: {y_i}")
    x_std = ( i - x_min) / (x_max - x_min)
    x_scaled = x_std * (x_max - x_min) + x_min
    print("Reproduced scale: ", x_std)
    print("output from minizinc ", predicted_flow[i])
print("Hi")


output_dict = convert_to_minizinc_data(model_source)

# dummy DZN export for now - if we have the encoding right, the rest is fancy-pants work
import json 
import os
json_file_name = os.path.join(os.path.dirname(__file__), 'nnet_data.json')
with open(json_file_name, "w") as out_file:
    json.dump(output_dict, out_file)

# now for the sink and source models
_, mzn_code_source = get_minizinc_nnet_parameters("source_net")
_, mzn_code_sink = get_minizinc_nnet_parameters("sink_net")

print(mzn_code_source)
print(mzn_code_sink)

json_file_name = os.path.join(os.path.dirname(__file__), 'nnet_data_sink_source.json')
output_dict_sink = convert_to_minizinc_data(model_sink, "sink_net")
output_dict_source = convert_to_minizinc_data(model_source, "source_net")
output_dict = output_dict_sink | output_dict_source
with open(json_file_name, "w") as out_file:
    json.dump(output_dict, out_file)


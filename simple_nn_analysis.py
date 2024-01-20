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
for i in range(0, 10): 
    y_i = model(th.tensor(scaler.transform([[i]]), dtype=th.float32)).detach().numpy().item()
    print(f"Time step {i}: Normalized {scaler.transform([[i]]).item()} Output: {y_i}")
    x_std = ( i - x_min) / (x_max - x_min)
    x_scaled = x_std * (x_max - x_min) + x_min
    print("Reproduced scale: ", x_std)
print("Hi")
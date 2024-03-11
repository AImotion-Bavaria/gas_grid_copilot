# Define the neural network
import torch as th
from torch import nn

hidden_units = 40
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, hidden_units )  # 1 input feature, 5 hidden neurons
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units , hidden_units )  # 5 hidden neurons, 1 output
        self.fc3 = nn.Linear(hidden_units , 1)  # 5 hidden neurons, 1 output 

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
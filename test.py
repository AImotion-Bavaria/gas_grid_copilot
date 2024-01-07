import numpy as np
import matplotlib.pyplot as plt

# Number of values
num_values = 10

def get_multimodal_flow(num_values, modes : list, max_flow : float):
    # Generate an array with 10 values between 0 and 10
    x = np.linspace(0, num_values, num_values)

    # Create Gaussian modes around the 3rd and 7th entries
    mode_functions = [np.exp(-(x - mode_center)**2 / (2 * 1**2)) for mode_center in modes]
    # Combine the modes and scale to get the final array with the highest value of 0.02
    combined = np.sum(mode_functions, axis=0)
    final_array = max_flow * (combined) / (np.max(combined))
    return x, final_array


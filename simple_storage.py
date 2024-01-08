import pandapipes as pp
import pandas as pd
import numpy as np

# plotting the state of charge
import matplotlib.pyplot as plt 

def get_example_line():
    # let's create the net first
    net = pp.create_empty_network(fluid="hydrogen")
    pn_bar = 30
    norm_temp = 293.15 # in K, would be 20° C

    #create junctions

    j = dict()
    for i in range(0, 4):
        j[i] = pp.create_junction(net, pn_bar=pn_bar, tfluid_k=norm_temp, name=f"Junction {i}")


    #create junction elements
    ext_grid = pp.create_ext_grid(net, junction=j[0], p_bar=pn_bar, t_k=293.15, name="Grid Connection 1")
    source = pp.create_source(net, junction=j[1], mdot_kg_per_s=0.2, name="My source")
    sink = pp.create_sink(net, junction=j[3], mdot_kg_per_s=0.01, name="My sink")
    pp.create_mass_storage(net, junction=j[2], mdot_kg_per_s=0.1, 
                           init_m_stored_kg=2, 
                           min_m_stored_kg=0, max_m_stored_kg=500,
                           name = "Test Storage",
                           type="Classical mass storage")

    # now for the actual pipes?
    #create branch elements
    pp.create_pipe_from_parameters(net, from_junction=j[0], to_junction=j[1], length_km = 10, diameter_m=0.4, name="Pipe 0")
    pp.create_pipe_from_parameters(net, from_junction=j[1], to_junction=j[2], length_km = 10, diameter_m=0.4, name="Pipe 1")
    pp.create_pipe_from_parameters(net, from_junction=j[2], to_junction=j[3], length_km = 20, diameter_m=0.4, name="Pipe 2")

    #valve1 = pp.create_valve(net, from_junction=j[1], to_junction=j[3], diameter_m=0.4, opened=True, name="Valve")
    #valve2 = pp.create_valve(net, from_junction=j[2], to_junction=j[4], diameter_m=0.4, opened=True, name="Valve")

    pp.pipeflow(net)

    # now for some of the results:
    return net

def get_multimodal_flow(num_values, modes : list, max_flow : float):
    # Generate an array with 10 values between 0 and 10
    x = np.linspace(0, num_values, num_values)

    # Create Gaussian modes around the 3rd and 7th entries
    mode_functions = [np.exp(-(x - mode_center)**2 / (2 * 1**2)) for mode_center in modes]
    # Combine the modes and scale to get the final array with the highest value of 0.02
    combined = np.sum(mode_functions, axis=0)
    final_array = max_flow * (combined) / (np.max(combined))
    return x, final_array

def plot_flow(x, y):
    # Plot the array
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title('Array with Gaussian Modes')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

def plot_trajectory(output_dict, reward_trajectory : pd.DataFrame = None):
    mass_df = output_dict["mass_storage.m_stored_kg"]
    ext_grid_flow = output_dict["res_ext_grid.mdot_kg_per_s"]
    source_flow = output_dict["res_source.mdot_kg_per_s"]
    sink_flow = output_dict["res_sink.mdot_kg_per_s"]
    mass_storage_flow = output_dict["mass_storage.mdot_kg_per_s"]
    mass_df.columns = ['mass_storage']
    
    # total external grid consumption:
    ext_grid_flow_total = np.sum(np.abs(ext_grid_flow))
    print("Consumed from external: ", ext_grid_flow_total)
    
    # Assuming the index of your data frames represents time
    time_index = mass_df.index

    # Create a horizontal subplot with 4 subplots
    if reward_trajectory is None:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            # Plotting the data frames on each subplot
        axs[0].plot(time_index, mass_df, color='blue')
        axs[0].set_title('Mass Storage')
        axs[0].set_xlabel('Values')
        axs[0].set_ylabel('Time')

        axs[1].plot(time_index, ext_grid_flow,  color='green', label ="Ext grid flow")
        axs[1].set_xlabel('Values')

        axs[1].plot( time_index, source_flow, color='red', label = "Source flow")
        axs[1].set_xlabel('Values')
        axs[1].legend()

        axs[1].plot( time_index, mass_storage_flow, color='blue', label = "Mass storage flow")
        axs[1].set_title('Flows')
        axs[1].set_xlabel('Values')

        axs[1].plot( time_index, sink_flow, label = "Sink flow")
        axs[1].set_xlabel('Values')
        axs[1].legend()
    else:
        fig, axs = plt.subplots(2, 2, figsize=(15, 8))
        # Plotting the data frames on each subplot
        axs[0,0].plot(time_index, mass_df, color='blue')
        axs[0,0].set_title('Mass Storage')
        axs[0,0].set_xlabel('Values')
        axs[0,0].set_ylabel('Time')

        axs[0, 1].plot(time_index, ext_grid_flow,  color='green', label ="Ext grid flow")
        axs[0, 1].set_xlabel('Values')

        axs[0, 1].plot( time_index, source_flow, color='red', label = "Source flow")
        axs[0, 1].set_xlabel('Values')
        axs[0, 1].legend()

        axs[0, 1].plot( time_index, mass_storage_flow, color='blue', label = "Mass storage flow")
        axs[0, 1].set_title('Flows')
        axs[0, 1].set_xlabel('Values')

        axs[0, 1].plot( time_index, sink_flow, label = "Sink flow")
        axs[0, 1].set_xlabel('Values')
        axs[0, 1].legend()

        reward_trajectory.plot(ax=axs[1,0]) 
   
        # Sum values for each column
        reward_sums = reward_trajectory.sum()

        # Create a bar plot for the specific subplot axs[0, 1]
        reward_sums.plot(kind='bar', ax=axs[1, 1])

        # Adding labels and title
        axs[1, 1].set_xlabel('Reward types')
        axs[1, 1].set_ylabel('Cumulative rewards')
        print("********")
        print(reward_sums)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    net = get_example_line()
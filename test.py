import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Define the range of quality values (replace these with your actual data)
def plot_quality(ax, min_quality = 0, max_quality = 100, curr_quality = 60):
    min_quality = 0
    max_quality = 100

    # Create a sample array of quality values
    quality_values = np.array([[curr_quality]])

    # Create a normalization object to map the values to the color map
    norm = Normalize(vmin=min_quality, vmax=max_quality)

    # Choose a colormap (e.g., 'RdYlGn' for Red-Yellow-Green)
    cmap = plt.get_cmap('RdYlGn')

    im = ax.imshow(quality_values, cmap=cmap, norm=norm)

    # Add a colorbar to show the mapping of values to colors
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', location="bottom", label='Q Values')

    
    cbar.ax.plot(curr_quality, 0.5, 'gx')
    cbar.ax.axvline(40, c='b', linewidth=4.0)

    # Set axis labels and title as needed
    ax.set_axis_off()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Create the plot
    fig, ax = plt.subplots()
    plot_quality(ax, 0, 100, 70)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as image
from matplotlib.widgets import Button, Slider
import matplotlib as mpl
import matplotlib.style as mplstyle
from matplotlib.gridspec import GridSpec
import os 

from simple_storage import plot_flows, plot_storage, plot_reward_trajectory

def update_plot(time_index, grid_image, plt_image, time_lines):
    plt_image.set_data(grid_image)
    for tl in time_lines:
        tl.set_xdata([time_index])

def image_plot(imgs, output_dict, reward_trajectory):
    gs=GridSpec(2,3) # 2 rows, 3 columns

    fig=plt.figure(figsize=(14,10))

    gs=GridSpec(2,3, height_ratios=[1,3]) # 2 rows, 3 columns
    ikigas_logo = os.path.join(os.path.dirname(__file__), 'ikigas.png')
    im = image.imread(ikigas_logo)

    ax1=fig.add_subplot(gs[0,0]) # First row, first column
    ax2=fig.add_subplot(gs[0,1]) # First row, second column
    ax3=fig.add_subplot(gs[0,2]) # First row, third column
    ax4=fig.add_subplot(gs[1,:]) # Second row, span all columns

    ikigas_logo = os.path.join(os.path.dirname(__file__), 'ikigas.png')
    im = image.imread(ikigas_logo)

    imax = ax4
    # remove ticks & the box from imax 
    imax.set_axis_off()
    plt_image = imax.imshow(imgs[5], aspect="equal")

    # adjust the main plot to make room for the sliders
    plt.tight_layout()
    fig.subplots_adjust(top = 0.85)

    axtime = fig.add_axes([0.25, 0.9, 0.65, 0.03])
    time_slider = Slider(
        ax=axtime,
        label='Time steps',
        valmin=0,
        valmax=len(imgs),
        valinit=5,
        valfmt='%0.0f',
        valstep=range(0,len(imgs))
    )

    # Create the Vertical lines on the histogram
    
    # ax3 should get the rewards
    # ax2 should get flow information
    plot_flows(output_dict, ax1)
    plot_storage(output_dict, ax2)
    plot_reward_trajectory(reward_trajectory, ax3)

    time_lines = [ax_.axvline(5, color='darkred') for ax_ in [ax1, ax2, ax3]]
    time_slider.on_changed(lambda t : update_plot(t, imgs[t], plt_image, time_lines ) )
    time_slider.set_val(5)

    # Logo business
    imax = fig.add_axes([0.05, 0.9, 0.1, 0.1])
    # remove ticks & the box from imax 
    imax.set_axis_off()
    # print the logo with aspect="equal" to avoid distorting the logo
    imax.imshow(im, aspect="equal")
    plt.show()

import pickle
if __name__ == "__main__":   
    mplstyle.use('fast')
    mpl.rcParams['path.simplify_threshold'] = 1.0
    mpl.rcParams['path.simplify'] = True
    # test with some images
    imgs_file_name = "SAC_trajectory.pickle"
    imgs_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), imgs_file_name)
    # using with statement
    with open(imgs_file_name, 'rb') as file:
        imgs = pickle.load(file)

    # also load the reward trajectory and output dict
    data_file_name = "SAC_trajectory_data.pickle"
    data_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_file_name)
    with open(data_file_name, 'rb') as file:
        content = pickle.load(file)
        print(content)
        output_dict, reward_trajectory = content
    image_plot(imgs, output_dict, reward_trajectory)
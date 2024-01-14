import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as image
from matplotlib.widgets import Button, Slider, RadioButtons
import matplotlib as mpl
import matplotlib.style as mplstyle
from matplotlib.gridspec import GridSpec
import os 

from simple_storage import plot_flows, plot_storage, plot_reward_trajectory

class IKIGasDemoPlot:
    def __init__(self) -> None:
        pass

    def image_plot(self, imgs, output_dict, reward_trajectory, plot_data : dict = None):
        mplstyle.use('fast')
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mpl.rcParams['path.simplify'] = True
        gs=GridSpec(2,3) # 2 rows, 3 columns

        fig=plt.figure(figsize=(15,11))
        gs=GridSpec(2,3, height_ratios=[1,3]) # 2 rows, 3 columns
        ikigas_logo = os.path.join(os.path.dirname(__file__), 'ikigas.png')
        im = image.imread(ikigas_logo)

        ax1=fig.add_subplot(gs[0,0]) # First row, first column
        ax2=fig.add_subplot(gs[0,1]) # First row, second column
        ax3=fig.add_subplot(gs[0,2]) # First row, third column
        ax4=fig.add_subplot(gs[1,:]) # Second row, span all columns
        self.axes = [ax1, ax2, ax3, ax4]

        ikigas_logo = os.path.join(os.path.dirname(__file__), 'ikigas.png')
        im = image.imread(ikigas_logo)

        imax = ax4
        # remove ticks & the box from imax 
        imax.set_axis_off()
        self.plt_image = imax.imshow(imgs[5], aspect="equal")

        # adjust the main plot to make room for the sliders
        plt.tight_layout()
        fig.subplots_adjust(top = 0.85, left = .06)

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

        # now the reward sliders 
        start_rewards = 0.15
        
        axreward_storage = fig.add_axes([0.15,  start_rewards, 0.2, 0.03])
        plt.gcf().text(0.15, start_rewards+0.05, "Reward priorities", fontsize=15, weight="bold")
        axreward_mass_flow = fig.add_axes([0.15, start_rewards - 0.05, 0.2, 0.03])
        axreward_difference = fig.add_axes([0.15, start_rewards - 0.1 , 0.2, 0.03])

        # [left, bottom, width, height] 
        reward_storage_slider = Slider(
            ax=axreward_storage,
            label='Reward storage',
            valmin=1,
            valmax=100,
            valinit=1,
            valfmt='%0.0f',
            valstep=[1, 10, 100]
        )
        
        reward_mass_flow_slider = Slider(
            ax=axreward_mass_flow,
            label='Reward mass flow',
            valmin=1,
            valmax=100,
            valinit=1,
            valfmt='%0.0f',
            valstep=[1, 10, 100]
        )

        reward_difference_slider = Slider(
            ax=axreward_difference,
            label='Reward difference',
            valmin=1,
            valmax=100,
            valinit=1,
            valfmt='%0.0f',
            valstep=[1, 10, 100]
        )
        # ax3 should get the rewards
        # ax2 should get flow information
        plot_flows(output_dict, ax1)
        self.storage_plot, self.storage_filler = plot_storage(output_dict, ax2)
        plot_reward_trajectory(reward_trajectory, ax3)

        # Create the Vertical lines on the histogram
        time_lines = [ax_.axvline(5, color='darkred') for ax_ in [ax1, ax2, ax3]]
        time_slider.on_changed(lambda t : self.update_plot(t, imgs[t], time_lines ) )
        time_slider.set_val(5)

        # all reward handlers
        for rw_slider in [reward_storage_slider, reward_mass_flow_slider, reward_difference_slider]:
            rw_slider.on_changed(lambda t : self.update_data( *plot_data[(reward_storage_slider.val, reward_mass_flow_slider.val, reward_difference_slider.val)] ))
        # Logo business
        imax = fig.add_axes([0.05, 0.9, 0.1, 0.1])
        # remove ticks & the box from imax 
        imax.set_axis_off()
        # print the logo with aspect="equal" to avoid distorting the logo
        imax.imshow(im, aspect="equal")
        plt.show()

    def update_plot(self, time_index, grid_image, time_lines):
        self.time_index = time_index
        self.plt_image.set_data(grid_image)
        for tl in time_lines:
            tl.set_xdata([time_index])

    def update_data(self, imgs, output_dict, reward_trajectory):
        print("Here: ")
        
        self.storage_plot.set_ydata(output_dict["mass_storage.m_stored_kg"])
        
        ax = self.storage_filler.axes
        self.storage_filler.remove()
        mass_storage_raw_vals = output_dict["mass_storage.m_stored_kg"][output_dict["mass_storage.m_stored_kg"].columns[0]]
        self.storage_filler = ax.fill_between(output_dict["mass_storage.m_stored_kg"].index, 0, mass_storage_raw_vals, alpha=0.2, color="royalblue")

        # reward_trajectory
        ax3 = self.axes[2]
        ax3.clear()
        plot_reward_trajectory(reward_trajectory, ax3)

        # the flows 
        ax1 = self.axes[0]
        ax1.clear()
        plot_flows(output_dict, ax1)
        
        # the images
        self.imgs = imgs
        self.plt_image.set_data(imgs[self.time_index])

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
    demo_plot = IKIGasDemoPlot()
    demo_plot.image_plot(imgs, output_dict, reward_trajectory)
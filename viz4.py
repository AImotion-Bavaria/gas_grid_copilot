import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as image
from matplotlib.widgets import Button, Slider, RadioButtons
import matplotlib as mpl
import matplotlib.style as mplstyle
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os 
import pandas as pd

from simple_storage import plot_flows, plot_storage, plot_reward_trajectory, plot_quality

class IKIGasDemoPlot:
    def __init__(self) -> None:
        pass

    def image_plot(self, imgs, output_dict, reward_trajectory, 
                   q_vals = None, 
                   plot_data : dict = None, 
                   all_reward_trajectories : pd.DataFrame = None):
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
        self.time_index = 5
        if q_vals:
            self.q_vals = [q_val.detach().numpy() for q_val in q_vals]
            q_vals = self.q_vals
        else:
            self.q_vals = q_vals

        imax = ax4
        # remove ticks & the box from imax 
        imax.set_axis_off()
        self.plt_image = imax.imshow(imgs[self.time_index], aspect="equal")

        # adjust the main plot to make room for the sliders
        plt.tight_layout()
        fig.subplots_adjust(top = 0.85, left = .06)

        axtime = fig.add_axes([0.25, 0.9, 0.65, 0.03])

        time_slider = Slider(
            ax=axtime,
            label='Time steps',
            valmin=0,
            valmax=len(imgs),
            valinit=self.time_index,
            valfmt='%0.0f',
            valstep=range(0,len(imgs))
        )

        # now the reward sliders 
        start_rewards = 0.15

        # q values 
        plt.gcf().text(0.55, start_rewards+0.05, "Current Q value (grid state)", fontsize=15, weight="bold")
        self.__axq_values = fig.add_axes([0.55,  0.04, 0.12, 0.12])
        if self.q_vals:
            self.q_plot, self.q_plot_cbar_marker, self.q_plot_cbar =  plot_quality(self.__axq_values , min_quality= np.min(q_vals).item(), max_quality=np.max(q_vals).item(), curr_quality=q_vals[self.time_index].item())
        else:
            self.q_plot, self.q_plot_cbar_marker, self.q_plot_cbar = plot_quality(self.__axq_values )

        
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
        self.reward_trajectory = reward_trajectory
        self.all_reward_trajectories = all_reward_trajectories

        # below we want to see the conflict interaction matrix
        axinteractionbutton = fig.add_axes([0.7, 0.55, 0.2, 0.03])
        interaction = Button(axinteractionbutton, 'Show model reward interactions')
        interaction.on_clicked(lambda val : self.show_interaction_matrix(self.reward_trajectory, "Reward interactions (over this model's traces)"))

        axallinteractionbutton = fig.add_axes([0.45, 0.55, 0.2, 0.03])     
        allinteraction = Button(axallinteractionbutton, 'Show all reward interactions')
        allinteraction.on_clicked(lambda val : self.show_interaction_matrix(self.all_reward_trajectories, "Reward interactions (over all traces)"))

        # Create the Vertical lines on the histogram
        self.time_lines = [ax_.axvline(5, color='darkred') for ax_ in [ax1, ax2, ax3]]
        time_slider.on_changed(lambda t : self.update_plot(t, imgs[t] ) )
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

    def show_interaction_matrix(self, interaction_matrix, title):
        plt.figure()
        corr = interaction_matrix.corr()
        ax = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap=sns.diverging_palette(10, 133, n=256, as_cmap=True))
        plt.title(title)
        #ax.set_aspect("auto")
        plt.tight_layout()
        plt.show()
    
    def update_plot(self, time_index, grid_image):
        self.time_index = time_index
        self.plt_image.set_data(grid_image)

        if self.q_vals:
            next_q_value = self.q_vals[time_index].item() 
            self.q_plot.set_data (np.array([[next_q_value]]))  
            self.q_plot_cbar_marker.set_data([[next_q_value, next_q_value], [0, 1]])

        for tl in self.time_lines:
            tl.set_xdata([time_index])

    def update_data(self, imgs, output_dict, reward_trajectory, q_vals):
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
        self.reward_trajectory = reward_trajectory

        # the flows 
        ax1 = self.axes[0]
        ax1.clear()
        plot_flows(output_dict, ax1)
        
        # the time lines 
        for tl in self.time_lines:
            tl.remove()

        self.time_lines = [ax_.axvline(self.time_index, color='darkred') for ax_ in [self.axes[i] for i in [0, 1, 2]]]

        # the images
        self.imgs = imgs
        self.plt_image.set_data(imgs[self.time_index])

        # the q_values 
        self.q_vals = [q_val.detach().numpy() for q_val in q_vals]
        q_vals = self.q_vals
        self.q_plot_cbar.remove()
        self.__axq_values.clear()
        self.q_plot, self.q_plot_cbar_marker, self.q_plot_cbar =  plot_quality(self.__axq_values , min_quality= np.min(q_vals).item(), max_quality=np.max(q_vals).item(), curr_quality=q_vals[self.time_index].item())
      

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
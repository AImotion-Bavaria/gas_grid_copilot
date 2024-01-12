from matplotlib.gridspec import GridSpec
import matplotlib.image as image
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style as mplstyle

from viz_step_by_step import plot_gas_network 
import simple_storage
import pandapipes as pp

from matplotlib.widgets import Button, Slider

line = simple_storage.get_example_line()
pp.pipeflow(line)
mplstyle.use('fast')
mpl.rcParams['path.simplify_threshold'] = 1.0
mpl.rcParams['path.simplify'] = True
fig=plt.figure(figsize=(14,7))

gs=GridSpec(2,3) # 2 rows, 3 columns

ax1=fig.add_subplot(gs[0,0]) # First row, first column
ax2=fig.add_subplot(gs[0,1]) # First row, second column
ax3=fig.add_subplot(gs[0,2]) # First row, third column
ax4=fig.add_subplot(gs[1,:]) # Second row, span all columns
import os

# Make a horizontal slider to control the frequency.



plot_gas_network(line, ax4)
ikigas_logo = os.path.join(os.path.dirname(__file__), 'ikigas.png')
im = image.imread(ikigas_logo)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, top = 0.85)

axtime = fig.add_axes([0.25, 0.9, 0.65, 0.03])
time_slider = Slider(
    ax=axtime,
    label='Time steps',
    valmin=0,
    valmax=10,
    valinit=5,
    valfmt='%0.0f',
    valstep=range(0,11)
)

time_slider.on_changed(lambda x : print("hi", x))
# put a new axes where you want the image to appear
# (x, y, width, height)
imax = fig.add_axes([0, 0.9, 0.1, 0.1])
# remove ticks & the box from imax 
imax.set_axis_off()
# print the logo with aspect="equal" to avoid distorting the logo
imax.imshow(im, aspect="equal")
plt.show()

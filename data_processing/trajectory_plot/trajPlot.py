import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml
import os

MAP_DIR = os.getcwd() + '/data_processing/trajectory_plot/maps/'
MAP_NAME = 'inb3235_small'
TRAJ_CSV = os.getcwd() + '/data_processing/bag_processing_bringup/traj/trajectory_interp.csv'
FONT_SIZE = 14

with open(MAP_DIR + MAP_NAME + '/map.yaml', 'r') as yaml_file:
    map_info = yaml.safe_load(yaml_file)

# Step 1: Load the PNG Image
map_image = mpimg.imread(MAP_DIR + MAP_NAME + '/map.pgm')

# Step 2: Plot the Map
# Get resolution and origin from YAML
resolution = map_info['resolution']
origin_x, origin_y = map_info['origin'][:2]

# Plot the map image
plt.imshow(map_image, extent=(origin_x, origin_x + len(map_image[0]) * resolution, 
                               origin_y, origin_y + len(map_image) * resolution),
           cmap='gray')

# Step 3: Load Trajectory Data
trajectory_data = pd.read_csv(TRAJ_CSV)

# Step 4: Plot Trajectories
plt.plot(trajectory_data['H1_X'].values, trajectory_data['H1_Y'].values, label='$H_1$')
plt.plot(trajectory_data['H2_X'].values, trajectory_data['H2_Y'].values, label='$H_2$')

# Add labels and legend
plt.xlabel('X [m]', fontsize=FONT_SIZE)
plt.ylabel('Y [m]', fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
# Set font size of ticks on both axes
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
# Set axis limits
plt.xlim(-1, 8.5)  # Set x-axis limits from 0 to 6
plt.ylim(-6, 4)  # Set y-axis limits from 0 to 12

# Show plot
plt.show()




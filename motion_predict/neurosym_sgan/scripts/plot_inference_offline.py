import os
import argparse
import time
import numpy as np
import inspect
import rospy
from contextlib import contextmanager
import subprocess
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml
import matplotlib.animation
import message_filters
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
#from sgan.losses import displacement_error as displacement_error_baseline
#from sgan.losses import final_displacement_error as final_displacement_error_baseline
from sgan.losses_informed import displacement_error, final_displacement_error
import torch
#import imageio.v2 as imageio

#from sgan.utils import int_tuple, bool_flag, get_total_norm
from std_msgs.msg import String,Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension

from darko_perception_msgs.msg import Humans, Human, HumansTrajs

plt.rcParams.update({'font.size': 20})


MODEL_NAME = "zara1"




# Define a custom legend handler
class AnyObjectHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create a single legend entry for all trajectories of the same type
        x = [0, width*2]  # Adjust width for spacing
        y = [(height - ydescent) / 2., (height - ydescent) / 2.]
        legline = plt.Line2D(x, y, linewidth=2, linestyle=orig_handle.get_linestyle())
        #legline = plt.Line2D(x, y, linewidth=2, color=orig_handle.get_color())
        return [legline]




class motpred_sub:

	def __init__(self, ):

		self.ade = []
		self.fde = []


	def plot_motion(self, trajs_gt, trajs_pred_neurosym, trajs_pred_baseline):

		print(len(trajs_pred_baseline.trajs))

		# to run GUI event loop
		#plt.ion()


		#plt.imshow(map_image, extent=(origin_x, origin_x + len(map_image[0]) * resolution, 
        #                       origin_y, origin_y + len(map_image) * resolution),cmap='gray')
		
		figure, ax = plt.subplots(figsize=(10, 10))
 

		# set each trajectory to a different color
		#cmap = plt.cm.autumn_r(np.linspace(0.1, 1, len(traj_gt_t.humans)))

		res = 1
		color = ['g', 'b', 'y', 'c', 'k', 'r']
		#cmap_small = ['g', 'b', 'y', 'c', 'k', 'r']
		#color = cm.tab20(np.linspace(0, 1, 200))
		humans_gt = {}
		humans_pred_neurosym = {}
		humans_pred_baseline = {}

		for dt_gt in range(len(trajs_gt.trajs)):
			traj_gt_t = trajs_gt.trajs[dt_gt]
			traj_pred_neurosym_t = trajs_pred_neurosym.trajs[dt_gt]
			traj_pred_baseline_t = trajs_pred_baseline.trajs[dt_gt]
			#if len(traj_gt_t.humans) != 0 and len(traj_pred_t.humans) != 0 :
			for dh in range(len(traj_gt_t.humans)):

				human_gt = traj_gt_t.humans[dh]
				human_gt_x = human_gt.centroid.pose.position.x
				human_gt_y = human_gt.centroid.pose.position.y
				human_gt_id = human_gt.id


				if human_gt_id in humans_gt.keys():
					humans_gt[human_gt_id].append([human_gt_x, human_gt_y])
				else:
					humans_gt[human_gt_id] = [[human_gt_x, human_gt_y]]

				#scatter1 = plt.scatter(human_gt_x, human_gt_y, color=cmap_small[human_gt_id], marker='o', s=100, label='', alpha=1, edgecolors=cmap_small[human_gt_id])
				#scatter1 = plt.scatter(human_gt_x, human_gt_y, color=color[res*human_gt_id], marker='o', s=100, label='', alpha=1, edgecolors=color[res*human_gt_id])


				if dt_gt == 0:
					plt.scatter(human_gt_x, human_gt_y, color=color[res*human_gt_id], marker='o', s=200, label='origin', alpha=1, edgecolors=color[res*human_gt_id])

			if dt_gt >= int(len(trajs_gt.trajs)/2) or dt_gt >= 0: 
				for dh in range(len(traj_pred_neurosym_t.humans)):

					human_pred = traj_pred_neurosym_t.humans[dh]
					human_pred_x = human_pred.centroid.pose.position.x
					human_pred_y = human_pred.centroid.pose.position.y
					human_pred_x = human_pred_x.cpu().detach().numpy()
					human_pred_y = human_pred_y.cpu().detach().numpy()
					human_pred_id = human_pred.id


					if human_pred_id in humans_pred_neurosym.keys():
						humans_pred_neurosym[human_pred_id].append([human_pred_x, human_pred_y])
					else:
						humans_pred_neurosym[human_pred_id] = [[human_pred_x, human_pred_y]]

					#scatter2 = plt.scatter(human_pred_x, human_pred_y, olor=color[res*human_pred_id], marker='x', s=100, label='', alpha=1, edgecolors='none')
					
					## init point for pred
					#if dt_gt == 0:
					#	plt.scatter(human_pred_x, human_pred_y, color='red', marker='o', s=200, label='origin', alpha=1, edgecolors='red')
		

				for dh in range(len(traj_pred_baseline_t.humans)):

					human_pred_base = traj_pred_baseline_t.humans[dh]
					human_pred_x_base = human_pred_base.centroid.pose.position.x
					human_pred_y_base = human_pred_base.centroid.pose.position.y
					human_pred_x_base = human_pred_x_base.cpu().detach().numpy()
					human_pred_y_base = human_pred_y_base.cpu().detach().numpy()
					human_pred_id_base = human_pred_base.id


					if human_pred_id_base in humans_pred_baseline.keys():
						humans_pred_baseline[human_pred_id_base].append([human_pred_x_base, human_pred_y_base])
					else:
						humans_pred_baseline[human_pred_id_base] = [[human_pred_x_base, human_pred_y_base]]

					#scatter3 = plt.scatter(human_pred_x_base, human_pred_y_base, color=color[res*human_pred_id_base], marker='*', s=100, label='', alpha=1, edgecolors=color[res*human_pred_id_base])
					
					## init point of pred
					#if dt_gt == 0:
					#	plt.scatter(human_pred_x_base, human_pred_y_base, color='green', marker='o', s=200, label='origin', alpha=1, edgecolors='green')




		humans_gt_values = list(humans_gt.values())
		humans_gt_values = np.stack(humans_gt_values)
		#print("-------------------", humans_gt_values.shape)
		humans_gt_values_trans = np.transpose(humans_gt_values, (1, 0, 2))
		humans_pred_neurosym_values = list(humans_pred_neurosym.values())
		humans_pred_neurosym_values = np.stack(humans_pred_neurosym_values)
		#print("-------------------", humans_pred_neurosym_values.shape)
		humans_pred_neurosym_values_trans = np.transpose(humans_pred_neurosym_values, (1, 0, 2))

		humans_pred_baseline_values = list(humans_pred_baseline.values())
		humans_pred_baseline_values = np.stack(humans_pred_baseline_values)
		#print("-------------------", humans_pred_baseline_values.shape)

		humans_pred_baseline_values_trans = np.transpose(humans_pred_baseline_values, (1, 0, 2))

		#print(humans_gt_values.shape[1])
		#print(humans_pred_neurosym_values.shape[1])
		#print(humans_pred_baseline_values.shape[1])

		ade_neurosym, fde_neurosym = self.eval_acc(torch.Tensor(humans_gt_values_trans[int((humans_gt_values_trans.shape[0])/2):, :, :]), torch.Tensor(humans_pred_neurosym_values_trans[int((humans_gt_values_trans.shape[0])/2):, :, :]))
		ade_baseline, fde_baseline = self.eval_acc(torch.Tensor(humans_gt_values_trans[int((humans_gt_values_trans.shape[0])/2):, :, :]), torch.Tensor(humans_pred_baseline_values_trans[int((humans_gt_values_trans.shape[0])/2):, :, :]))
		#print("ade =", ade.item())
		#print("fde =", fde.item())
		#self.ade.append(ade.item())
		#self.fde.append(fde.item())
		#rospy.loginfo("average ade over runtime %.4f", sum(self.ade)/len(self.ade))
		#rospy.loginfo("average fde over runtime %.4f", sum(self.fde)/len(self.fde))


		# Plot trajectories
		for el in humans_gt.keys():
			plt1, = plt.plot(humans_gt_values[el-1][:,0], humans_gt_values[el-1][:,1], color=color[res*el], linestyle='-', linewidth=4, label='GT')
			plt2, = plt.plot(humans_pred_neurosym_values[el-1][int((humans_pred_neurosym_values.shape[1])/2):, 0], humans_pred_neurosym_values[el-1][int((humans_pred_neurosym_values.shape[1])/2):, 1], linestyle=':', color=color[res*el], linewidth=4, label='Pred-neurosym')
			plt3, = plt.plot(humans_pred_baseline_values[el-1][int((humans_pred_baseline_values.shape[1])/2):, 0], humans_pred_baseline_values[el-1][int((humans_pred_baseline_values.shape[1])/2):, 1], linestyle='-.', color=color[res*el], linewidth=4, label='Pred-baseline')


		# # Plot trajectories
		# for el in humans_gt.keys():
		# 	plt1, = plt.plot(humans_gt_values[el-1][:,0], humans_gt_values[el-1][:,1], color=color[res*el], linestyle='-', linewidth=4, label='GT')
		# 	plt2, = plt.plot(humans_pred_neurosym_values[el-1][:, 0], humans_pred_neurosym_values[el-1][:, 1], linestyle=':', color=color[res*el], linewidth=4, label='Pred-neurosym')
		# 	plt3, = plt.plot(humans_pred_baseline_values[el-1][:, 0], humans_pred_baseline_values[el-1][:, 1], linestyle='-.', color=color[res*el], linewidth=4, label='Pred-baseline')



		plt.xlabel('X [m]', fontsize=20)
		plt.ylabel('Y [m]', fontsize=20)
		plt.title('$ADE^{sym}$=%s, $FDE^{sym}$=%s \n $ADE^b$=%s, $FDE^b$=%s'%(round(ade_neurosym.item(),3),round(fde_neurosym.item(),3), round(ade_baseline.item(),3),round(fde_baseline.item(),3)))

		#ax.legend((plt1, plt2, plt3), ('GT', 'Prediction-neurosym', 'Prediction-baseline'), scatterpoints=1, loc='',ncol=1,fontsize=20)

		## Method 1
		#plt.legend(handler_map={plt1: AnyObjectHandler(), plt2: AnyObjectHandler(), plt3: AnyObjectHandler()})

		## Method 2
		# Dictionary to keep track of line styles added to legend
		added_line_styles = {}

		# Get all lines on the current plot
		lines = plt.gca().get_lines()

		# Iterate through lines and add only one legend item per linestyle based on line style
		for line in lines:
		    line_style = line.get_linestyle()
		    if line_style not in added_line_styles:
		        added_line_styles[line_style] = line


		# Add legend items based on line style
		legend_handles = [Line2D([0], [0], linestyle=line.get_linestyle(), color='k') for line in added_line_styles.values()]
		legend_labels = [f'{line.get_label()}' for line in added_line_styles.values()]

		## Adding legend with custom legend handles and labels
		plt.legend(legend_handles, legend_labels)
		#plt.legend(legend_handles, legend_labels, bbox_to_anchor=(1.01, -0.13), borderaxespad=0., ncol=3)
		#plt.legend(legend_handles, legend_labels, bbox_to_anchor=(0.65, 0.98), loc='upper left', borderaxespad=0.)
		
		#plt.subplots_adjust(bottom=0.2)

		## Add a legend
		#legend = plt.legend()

		## Change the colors of legend handles		
		#for handle in legend_handles:
		#    handle.set_color('black')  

		#leg = ax.get_legend()
		#[lgd.set_color('black') for lgd in leg.legendHandles]
		plt.axis('equal')
		
		# Set the same number of ticks for both axes automatically
		#plt.locator_params(nbins=5)

		# Set equal size of ticks difference
		#num_ticks = 4
		#plt.locator_params(axis='x', nbins=num_ticks)
		#plt.locator_params(axis='y', nbins=num_ticks)
				
		plt.show()



	def eval_acc(self, traj_gt, traj_pred):
		loss_ade, loss_std = displacement_error(traj_pred, traj_gt, mode='raw')
		loss_fde  = final_displacement_error(traj_pred[-1], traj_gt[-1], mode='raw')
		return torch.sum(loss_ade, dim=0),torch.sum(loss_fde, dim=0)



if __name__ == '__main__':
	print("==================================================Get INPUT FROM DARKO WP5 ==================================================")
	plt_interface = motpred_sub() 

	## investigate GT data if no combined models
	# data_perception1_thor_baseline = np.load('traj/thor/thor_neurosym_bag/data_perception1_thor_baseline.npy', allow_pickle=True)
	# data_perception2_thor_baseline = np.load('traj/thor/thor_neurosym_bag/data_perception2_thor_baseline.npy', allow_pickle=True)
	# data_perception1_thor_neurosym = np.load('traj/thor/thor_neurosym_bag/data_perception1_thor_neurosym.npy', allow_pickle=True)
	# data_perception2_thor_neurosym = np.load('traj/thor/thor_neurosym_bag/data_perception2_thor_neurosym.npy', allow_pickle=True)

	# print("GT base 1 ", data_perception2_thor_baseline[0])
	# print("GT neurosym 1", data_perception2_thor_neurosym[0])

	# combined models
	if MODEL_NAME == "thor":
		data_perception1 = np.load('traj/thor/thor_neurosym_bag/combined/data_perception1_thor_neurosym.npy', allow_pickle=True)
		data_perception2 = np.load('traj/thor/thor_neurosym_bag/combined/data_perception2_thor_neurosym.npy', allow_pickle=True)
		data_perception3 = np.load('traj/thor/thor_neurosym_bag/combined/data_perception3_thor_baseline.npy', allow_pickle=True)

	else:
		data_perception1 = np.load('traj/zara1/zara1_baseline_bag/combined/data_perception1_zara1_neurosym.npy', allow_pickle=True)
		data_perception2 = np.load('traj/zara1/zara1_baseline_bag/combined/data_perception2_zara1_neurosym.npy', allow_pickle=True)
		data_perception3 = np.load('traj/zara1/zara1_baseline_bag/combined/data_perception3_zara1_baseline.npy', allow_pickle=True)

	rospy.loginfo("--------------------------- Starting to plot the data -----------------------")
	for l in range(len(data_perception2)):
		rospy.loginfo("---------------------------     -----------------------")

		ind_neurosym = np.where(np.array([el.header.stamp for el in data_perception1]) == data_perception2[l].header.stamp)[0].tolist()
		ind_baseline = np.where(np.array([el.header.stamp for el in data_perception3]) == data_perception2[l].header.stamp)[0].tolist()
		

		if len(ind_neurosym)!= 0 :
			print("ind_neurosym=", ind_neurosym)
			plt_interface.plot_motion(data_perception2[l], data_perception1[ind_neurosym[0]-1], data_perception3[ind_neurosym[0]-1]) # every 2 published predictions we get 1 published GT





	


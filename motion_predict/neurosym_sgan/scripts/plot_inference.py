import os
import argparse
import time
import numpy as np
import inspect
import rospy
from contextlib import contextmanager
import subprocess
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml
import matplotlib.animation
import message_filters
import matplotlib.colors as mcolors
from matplotlib.pyplot import cm
#from sgan.losses import displacement_error, final_displacement_error
from sgan.losses_informed import displacement_error, final_displacement_error
import torch
#import imageio.v2 as imageio

#from sgan.utils import int_tuple, bool_flag, get_total_norm
from std_msgs.msg import String,Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension

from darko_perception_msgs.msg import Humans, Human, HumansTrajs

plt.rcParams.update({'font.size': 20})

# MAP_DIR = os.path.expanduser('~/DARKO/src/motion_predict/qtc_sgan/maps/')
# MAP_NAME = 'inb3235_small'


# with open(MAP_DIR + MAP_NAME + '/map.yaml', 'r') as yaml_file:
#     map_info = yaml.safe_load(yaml_file)

# # Step 1: Load the PNG Image
# map_image = mpimg.imread(MAP_DIR + MAP_NAME + '/map.pgm')

# resolution = map_info['resolution']
# origin_x, origin_y = map_info['origin'][:2]


class motpred_sub:

	def __init__(self, ):

		rospy.init_node('darko_motion_plot', anonymous=True)
		self.rate = rospy.Rate(10)
		self.humans = 1
		self.data_pred = []
		self.data_gt = []
		#self.start = rospy.get_rostime()

		self.subpred1 = rospy.Subscriber('/hri/human_poses_prediction', HumansTrajs, self.sub1_callback)
		self.subgt2 = rospy.Subscriber('/hri/human_poses_gt', HumansTrajs, self.sub2_callback)
		#self.sync_sub = message_filters.ApproximateTimeSynchronizer([self.subpred1, self.subgt2], queue_size =1000, slop=0.2)
		#self.sync_sub.registerCallback(self.subsync_callback)

		self.min_ped = 1
		self.pred_stamp = 0
		self.seq = 0
		self.s = 5 # sequences
		self.ade = []
		self.fde = []
		self.seq_data = [] 


	# def subsync_callback(self, data_Pred, data_GT):

	# 	rospy.loginfo("starting data collection")

	# 	if data_Pred.header.seq != data_GT.header.seq:
	# 		rospy.loginfo("starting data collection")
	# 		self.data_pred = data_Pred
	# 		while self.data_pred.header.stamp != data_GT.header.stamp:
	# 			print("inner data_GT.header.seq= ", data_GT.header.stamp)
	# 			print("inner data_pred.header.seq = ", self.data_pred.header.stamp)
	# 			#rospy.sleep(0.01)
	# 		self.data_gt = data_GT
	# 		rospy.loginfo("--------------------------- Starting to plot the data -----------------------")
	# 		self.plot_motion(self.data_gt, self.data_pred)
	# 		rospy.sleep(1)
	# 	else:
	# 		print("else data_GT.header.stamp = ", data_GT.header.seq)
	# 		print("else data_pred.header.stamp = ", self.data_pred.header.seq)
	# 		self.data_pred = data_Pred
	# 		self.data_gt = data_GT
	# 		rospy.loginfo("--------------------------- Starting to plot the data -----------------------")
	# 		self.plot_motion(self.data_gt, self.data_pred)
	# 		rospy.sleep(1)




	# def sub1_callback(self, data_pred1):

	# 	rospy.loginfo("starting Pred data collection")
		
	# 	if 	(self.seq == 0):
	# 		self.seq = 1
	# 		self.data_pred = data_pred1
	# 		while self.data_pred.header.stamp != self.data_gt.header.stamp:
	# 			print("------------------------in while----------------------")
	# 			#rospy.sleep(0.01)
	# 		rospy.loginfo("--------------------------- Starting to plot the data -----------------------")
	# 		self.plot_motion(self.data_gt, self.data_pred)
	# 		rospy.sleep(1)
	# 		self.seq = 0



	def sub1_callback(self, data_pred1):

		rospy.loginfo("starting Pred data collection")
		
		if 	len(self.data_pred) < self.s:

			#rospy.loginfo("here")

			self.data_pred.append(data_pred1)
			#rospy.loginfo("there")

			if len(self.data_pred) == self.s: # and len(self.data_gt) == self.s): #or  (len(self.data_pred) >= self.s and  len(self.data_gt) == self.s) or (len(self.data_pred) == self.s and  len(self.data_gt) >= self.s) :

				rospy.loginfo("--------------------------- Starting to plot the data -----------------------")
				for l in range(len(self.data_gt)):
					rospy.loginfo("---------------------------     -----------------------")
					#print("data gt header = ", [el.header.stamp for el in self.data_gt], "l=", l)
					#print("data pred header = ", [el.header.stamp for el in self.data_pred], "l=", l)
					ind = np.where(np.array([el.header.stamp for el in self.data_pred]) == self.data_gt[l].header.stamp)[0].tolist()
					
					if len(ind)!= 0:
						print("ind=", ind)
						self.plot_motion(self.data_gt[l], self.data_pred[ind[0]-1]) # every 2 published predictions we get 1 published GT
						rospy.sleep(1)

				self.data_pred = []
				self.data_gt = []



	def sub2_callback(self, data_gt2):
		rospy.loginfo("starting GT data collection")
        
		if len(self.data_gt) < self.s:
			self.data_gt.append(data_gt2)



	def plot_motion(self, trajs_gt, trajs_pred):

		# to run GUI event loop
		#plt.ion()


		#plt.imshow(map_image, extent=(origin_x, origin_x + len(map_image[0]) * resolution, 
        #                       origin_y, origin_y + len(map_image) * resolution),cmap='gray')
		
		figure, ax = plt.subplots(figsize=(12, 12))
 

		# set each trajectory to a different color
		#cmap = plt.cm.autumn_r(np.linspace(0.1, 1, len(traj_gt_t.humans)))

		cmap_small = ['g', 'b', 'y', 'c', 'k', 'r']
		color = cm.rainbow(np.linspace(0, 1, 200))
		cmap = mcolors.CSS4_COLORS
		#cvalues = np.random.randint(0, 200, size=200)
		#norm = plt.Normalize(0, 200)
		humans_gt = {}
		humans_pred = {}

		for dt_gt in range(len(trajs_gt.trajs)):
			traj_gt_t = trajs_gt.trajs[dt_gt]
			traj_pred_t = trajs_pred.trajs[dt_gt]
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

				#scatter1 = plt.scatter(human_gt_x, human_gt_y, c=cmap[human_gt_id], marker='o', s=100, label='', alpha=1, edgecolors=cmap[human_gt_id])
				#scatter1 = plt.scatter(human_gt_x, human_gt_y, c=human_gt_id, marker='o', s=100, label='', alpha=1, cmap ='viridis', norm = norm)
				scatter1 = plt.scatter(human_gt_x, human_gt_y, c=color[human_gt_id], marker='o', s=100, label='', alpha=1, edgecolors=color[human_gt_id])

				if dt_gt == 0:
					#plt.scatter(human_gt_x, human_gt_y, c=cmap[human_gt_id], marker='o', s=200, label='origin', alpha=1, cmap ='viridis', norm = norm)
					plt.scatter(human_gt_x, human_gt_y, c=color[human_gt_id], marker='o', s=200, label='origin', alpha=1, edgecolors=color[human_gt_id])

			if dt_gt >= int(len(trajs_gt.trajs)/2):
				for dh in range(len(traj_pred_t.humans)):

					human_pred = traj_pred_t.humans[dh]
					human_pred_x = human_pred.centroid.pose.position.x
					human_pred_y = human_pred.centroid.pose.position.y
					human_pred_id = human_pred.id


					if human_pred_id in humans_pred.keys():
						humans_pred[human_pred_id].append([human_pred_x, human_pred_y])
					else:
						humans_pred[human_pred_id] = [[human_pred_x, human_pred_y]]

					scatter2 = plt.scatter(human_pred_x, human_pred_y, c=color[human_pred_id], marker='x', s=100, label='', alpha=1, edgecolors='none')
		


		# humans_gt_values = list(humans_gt.values())
		# humans_gt_values = np.stack(humans_gt_values)
		# #print("-------------------", humans_gt_values.shape)
		# humans_gt_values = np.transpose(humans_gt_values, (1, 0, 2))
		# humans_pred_values = list(humans_pred.values())
		# humans_pred_values = np.stack(humans_pred_values)
		# #print("-------------------", humans_pred_values.shape)
		# humans_pred_values = np.transpose(humans_pred_values, (1, 0, 2))
		
		# #print("-------------------", humans_gt_values.shape)

		# ade, fde = self.eval_acc(torch.Tensor(humans_gt_values[int((humans_gt_values.shape[0])/2):, :, :]), torch.Tensor(humans_pred_values))
		# print("ade =", ade.item())
		# print("fde =", fde.item())
		# self.ade.append(ade.item())
		# self.fde.append(fde.item())
		# rospy.loginfo("average ade over runtime %.4f", sum(self.ade)/len(self.ade))
		# rospy.loginfo("average fde over runtime %.4f", sum(self.fde)/len(self.fde))


		plt.xlabel('X [m]', fontsize=20)
		plt.ylabel('Y [m]', fontsize=20)
		#plt.title('ADE=%s, FDE=%s'%(round(ade.item(),3),round(fde.item(),3)))
		ax.legend((scatter1, scatter2), ('GT', 'Prediction'), scatterpoints=1, loc='best',ncol=1,fontsize=20, title= "Trajectries")
		
		## Add a legend
		#legend = plt.legend()
		# Change the colors of legend handles
		#for handle in legend.legendHandles:
		#    handle.set_color('black')  

		leg = ax.get_legend()
		[lgd.set_color('black') for lgd in leg.legendHandles]
		
		#labs =['ADE=%s'%(round(ade.item(),3)), 'FDE=%s'%(round(fde.item(),3))]
		#ax2 =ax.twinx()
		#ax2.legend((scatter1, scatter2),labs,  scatterpoints=1,loc='upper left', fontsize=12)

		# Retrieve legend location
		#legend = plt.legend()
		#legend_pos = legend.get_window_extent()
		#print(legend_pos)
		#plt.text(origin_x+2, origin_y+4.5, 'ADE=%s'%(round(ade.item(),3)), horizontalalignment='center', verticalalignment='center', fontsize=20)
		#plt.text(origin_x+2, origin_y+4,'FDE=%s'%(round(fde.item(),3)), horizontalalignment='center', verticalalignment='center', fontsize=20)
		# produce a legend with the unique colors from the scatter
		#kw1 = dict(num= ["GT", "Prediction"])
		#kw2 = dict(num= ["GT", "Prediction"])
		#legend = ax.legend([*scatter1.legend_elements(**kw1),*scatter2.legend_elements(**kw2)] ,loc="upper right", title="")

		# set limits
		#plt.xlim(0,8) 
		#plt.ylim(-4,0)

		#plt.axis('equal')
		plt.show()
		#self.seq_data.append([trajs_gt, trajs_pred, ade, fde])


	def eval_acc(self, traj_gt, traj_pred):
		loss_ade, loss_std = displacement_error(traj_pred, traj_gt, mode='raw')
		loss_fde  = final_displacement_error(traj_pred[-1], traj_gt[-1], mode='raw')
		return torch.sum(loss_ade, dim=0),torch.sum(loss_fde, dim=0)


if __name__ == '__main__':
	print("==================================================Get INPUT FROM DARKO WP5 ==================================================")
	plt_interface = motpred_sub() 
	start = time.time()


	while not rospy.is_shutdown() and (time.time()-start <= 130):
		plt_interface.rate.sleep()
	np.save('seq_data_neurosym_thor.npy', np.array(plt_interface.seq_data, dtype=object), allow_pickle=True)
		#print("Elapsed time ==============", time.time()-start)

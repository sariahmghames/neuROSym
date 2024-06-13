import argparse
import os
import torch
import itertools
import rospy
import numpy as np

from attrdict import AttrDict

from sgan.data.loader_informed import data_loader
from sgan.models_informed import TrajectoryGenerator
from sgan.losses_informed import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from sgan.utils import int_tuple, bool_flag, get_total_norm
from std_msgs.msg import String,Int32,Int32MultiArray,MultiArrayLayout,MultiArrayDimension
from people_msgs.msg import People

from darko_perception_msgs.msg import Humans, Human, HumansTrajs
from ptflops import get_model_complexity_info


from sgan.data.qtc import qtcc1

qtc_dic =  {'[-1, -1, -1, -1]': '0.0625', '[-1, -1, -1, 0]': '0.0417', '[-1, -1, -1, 1]': '0.0625', '[-1, -1, 0, -1]': '0.0417', '[-1, -1, 0, 0]': '0.0278', '[-1, -1, 0, 1]': '0.0417', '[-1, -1, 1, -1]': '0.0625', '[-1, -1, 1, 0]': '0.0417', '[-1, -1, 1, 1]': '0.0625', '[-1, 0, -1, -1]': '0.0417', '[-1, 0, -1, 0]': '0.0278', '[-1, 0, -1, 1]': '0.0417', '[-1, 0, 0, -1]': '0.0278', '[-1, 0, 0, 0]': '0.0185', '[-1, 0, 0, 1]': '0.0278', '[-1, 0, 1, -1]': '0.0417', '[-1, 0, 1, 0]': '0.0278', '[-1, 0, 1, 1]': '0.0417', '[-1, 1, -1, -1]': '0.0625', '[-1, 1, -1, 0]': '0.0417', '[-1, 1, -1, 1]': '0.0625', '[-1, 1, 0, -1]': '0.0417', '[-1, 1, 0, 0]': '0.0278', '[-1, 1, 0, 1]': '0.0417', '[-1, 1, 1, -1]': '0.0625', '[-1, 1, 1, 0]': '0.0417', '[-1, 1, 1, 1]': '0.0625', '[0, -1, -1, -1]': '0.0417', '[0, -1, -1, 0]': '0.0278', '[0, -1, -1, 1]': '0.0417', '[0, -1, 0, -1]': '0.0278', '[0, -1, 0, 0]': '0.0185', '[0, -1, 0, 1]': '0.0278', '[0, -1, 1, -1]': '0.0417', '[0, -1, 1, 0]': '0.0278', '[0, -1, 1, 1]': '0.0417', '[0, 0, -1, -1]': '0.0278', '[0, 0, -1, 0]': '0.0185', '[0, 0, -1, 1]': '0.0278', '[0, 0, 0, -1]': '0.0185', '[0, 0, 0, 0]': '0.0123', '[0, 0, 0, 1]': '0.0185', '[0, 0, 1, -1]': '0.0278', '[0, 0, 1, 0]': '0.0185', '[0, 0, 1, 1]': '0.0278', '[0, 1, -1, -1]': '0.0417', '[0, 1, -1, 0]': '0.0278', '[0, 1, -1, 1]': '0.0417', '[0, 1, 0, -1]': '0.0278', '[0, 1, 0, 0]': '0.0185', '[0, 1, 0, 1]': '0.0278', '[0, 1, 1, -1]': '0.0417', '[0, 1, 1, 0]': '0.0278', '[0, 1, 1, 1]': '0.0417', '[1, -1, -1, -1]': '0.0625', '[1, -1, -1, 0]': '0.0417', '[1, -1, -1, 1]': '0.0625', '[1, -1, 0, -1]': '0.0417', '[1, -1, 0, 0]': '0.0278', '[1, -1, 0, 1]': '0.0417', '[1, -1, 1, -1]': '0.0625', '[1, -1, 1, 0]': '0.0417', '[1, -1, 1, 1]': '0.0625', '[1, 0, -1, -1]': '0.0417', '[1, 0, -1, 0]': '0.0278', '[1, 0, -1, 1]': '0.0417', '[1, 0, 0, -1]': '0.0278', '[1, 0, 0, 0]': '0.0185', '[1, 0, 0, 1]': '0.0278', '[1, 0, 1, -1]': '0.0417', '[1, 0, 1, 0]': '0.0278', '[1, 0, 1, 1]': '0.0417', '[1, 1, -1, -1]': '0.0625', '[1, 1, -1, 0]': '0.0417', '[1, 1, -1, 1]': '0.0625', '[1, 1, 0, -1]': '0.0417', '[1, 1, 0, 0]': '0.0278', '[1, 1, 0, 1]': '0.0417', '[1, 1, 1, -1]': '0.0625', '[1, 1, 1, 0]': '0.0417', '[1, 1, 1, 1]': '0.0625', '[10, 10, 10, 10]': '0.0123'}


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--save', default=1, type=bool_flag)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--pooling_type', default='pool_net')




class motpred_pub:

	def __init__(self, args):

		rospy.init_node('darko_motion_pred', anonymous=True)
		self.rate = rospy.Rate(1)
		self.humans = 1
		self.message_count_1 = 0
		self.message_count_2 = 0
		self.data_perception1 = []
		self.data_perception2 = []
		self.start = rospy.get_rostime()


		self.pubNeuroSyM = rospy.Publisher('/hri/human_poses_prediction', HumansTrajs, queue_size=50)
		self.pubNeuroGT = rospy.Publisher('/hri/human_poses_gt', HumansTrajs, queue_size=50)

		self.subperception1 = rospy.Subscriber('/perception/humans', Humans, self.sub1_callback)
		self.subperception2 = rospy.Subscriber('/perception/humans', Humans, self.sub2_callback)

		self.obs_motion = []
		self.args = args
		self.obs_len = args.obs_len
		self.pred_len = args.pred_len
		self.seq_len = self.obs_len + self.pred_len
		self.repeat = 2
		self.labels_dir = "sgan/data/" 
		self.filename = "cnd_labels.txt" 
		self.min_ped = 1
		self.seq_start_end = []
		self.pred_stamp = 0
		self.inference_time = 0.0
		self.batches = 0



	def Trajectory(self, data):

		seq_list = [] #  each row is 1 sequence and each sequence is len(seq_len)=nb_frames/seq and in each frame there is nb of peds considered then each x and y 
		seq_list_weight = [] 
		seq_list_rel = [] 
		ped_ids = []


		frames = np.unique(data[:, 0]).tolist()
		frame_data = []
		for frame in frames:
		    frame_data.append(data[frame == data[:, 0], :])

		curr_seq_data = np.concatenate(frame_data[0: self.obs_len], axis=0)
		peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
		#print("peds in curr seq=", peds_in_curr_seq)
		curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,self.obs_len))
		curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.obs_len))
		curr_seq_qtc_AB = []

		num_peds_considered = 0
		print("number of peds ====================", len(peds_in_curr_seq))
        
		for _, ped_id in enumerate(peds_in_curr_seq): 

		    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==ped_id, :]
		    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
		    # Check if the seq len is right
		    pad_front = frames.index(curr_ped_seq[0, 0])  
		    pad_end = frames.index(curr_ped_seq[-1, 0]) + 1

		    if ((pad_end - pad_front != self.obs_len) or (curr_ped_seq.shape[0]!= self.obs_len)):
		        continue
		    
		    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) # (2: are x and y) so final shape = 2 x seq len
		    curr_ped_seq = curr_ped_seq # shape : 2 x obs len
		    
		    # Make coordinates relative ("with respect to frames")
		    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
		    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1] # current - prev
		    _idx = num_peds_considered
		    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
		    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
		    # Linear vs Non-Linear Trajectory
		    #_non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
		    #curr_loss_mask[_idx, pad_front:pad_end] = 1
		    num_peds_considered += 1 # the num of peds that appear in all the frames considered for the seq len (so in all the 16 frames)
		    ped_ids.append(ped_id)
        

		ind_excl = [(x*num_peds_considered)+x for x in range(num_peds_considered)]
		idx_qtc_weight = 0
		peds_comb_list = list(itertools.product(curr_seq[:num_peds_considered], repeat=self.repeat))

		if (self.repeat <= num_peds_considered and num_peds_considered!=1):
		    #total_sequences +=1
		    curr_seq_weight = torch.ones((num_peds_considered * num_peds_considered, 2, self.obs_len)) # weight are copied (same) along dim = 1, 

		    for tup_idx, tup in enumerate(peds_comb_list):
		        if tup_idx not in ind_excl:
		            curr_seq_qtc_AB = qtcc1(tup[0][:,:self.obs_len].transpose(), tup[1][:,:self.obs_len].transpose(), qbits = 4)
		            labels = [qtc_dic[str(list(x))] for x in curr_seq_qtc_AB]
		            curr_seq_weight[idx_qtc_weight, 0, :(self.obs_len)] = torch.tensor([[0]+[float(i) for i in labels]]) # 0 because no weight for t0, given no state was before it
		            curr_seq_weight[idx_qtc_weight, 1, :(self.obs_len)] = torch.tensor([[0]+[float(i) for i in labels]])
		            idx_qtc_weight += 1
		        else:
		            idx_qtc_weight += 1

		else:
		    curr_seq_weight = torch.zeros((1 , 2, self.obs_len))


		if num_peds_considered >= self.min_ped:
		    print("num_peds_considered = ", num_peds_considered)
		    seq_list.append(curr_seq[:num_peds_considered])
		    seq_list_weight.append(curr_seq_weight[:(num_peds_considered*num_peds_considered)])
		    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        
		seq_list = np.concatenate(seq_list, axis=0) # num_sequences, nb peds , seq_len, 2 # num sequences are batched
		seq_list_rel = np.concatenate(seq_list_rel, axis=0)


		# Convert numpy -> Torch Tensor
		obs_traj_curr = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float).permute(2, 0, 1) # shape : num_sequences, 2 , seq_len, (before permute)
		obs_traj_weight_curr = seq_list_weight # shape : num_sequences, 2 , seq_len,
		obs_traj_rel_curr = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float).permute(2, 0, 1)


		#print("obs traj shape = ", obs_traj_curr.size())
		assert len(obs_traj_curr.size()) == 3
		assert len(obs_traj_rel_curr.size()) == 3

		return obs_traj_curr.cuda(), obs_traj_rel_curr.cuda(), obs_traj_weight_curr, ped_ids, num_peds_considered



	def sub1_callback(self, data_percep1):

		if self.message_count_1 < self.obs_len:
			#rospy.loginfo("restarting collecting data 1")
			self.data_perception1.append([[data_percep1.header.stamp.nsecs, int(ped.name), ped.position.x, ped.position.y] for ped in data_percep1.people])
			self.message_count_1 += 1
			#rospy.loginfo("nb of messages collected = %d", self.message_count_1)

			if self.message_count_1 == self.obs_len:
				self.process_received_percep1()


	def process_received_percep1(self,):
		self.batches+=1
		# reshaping obs_motion_curr into pedsx2xtraj_len
		obs_traj = np.concatenate(self.data_perception1, axis=0)
		#future_gt_traj = np.concatenate(obs_motion_curr[self.obs_len, :], axis=0)
        
		obs_traj_curr, obs_traj_rel_curr, obs_traj_weight_curr, ped_ids, num_peds_considered = self.Trajectory(obs_traj)
		
		path = args.model_path
		checkpoint = torch.load(path)
		#_args = AttrDict(checkpoint['args'])

		self.seq_start_end=[(0,num_peds_considered)]
		generator = self.get_generator(checkpoint)
		pred_traj_fake_rel = generator(obs_traj_curr, obs_traj_rel_curr, obs_traj_weight_curr, self.seq_start_end) 
		pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj_curr[-1])


		full_pred_traj = torch.cat((obs_traj_curr,pred_traj_fake), dim=0)
		
		pred_trajs = HumansTrajs()
		pred_trajs.header.stamp = rospy.Time.now()
		self.pred_stamp = pred_trajs.header.stamp
		print("pred stamp = ", pred_trajs.header.stamp)
		pred_trajs.trajs = []

		for t in range(self.seq_len):
			# create a msg to publish the predicted traj
			pred_traj = Humans()

			pred_traj.header.stamp = rospy.Time.now()

			pred_traj.humans = []
			for h in range(full_pred_traj[t].size(0)):
				ped = Human()
				ped.id = int(ped_ids[h])
				ped.centroid.pose.position.x = full_pred_traj[t][h][0]
				ped.centroid.pose.position.y = full_pred_traj[t][h][1]
				pred_traj.humans.append(ped)
			pred_trajs.trajs.append(pred_traj)
		self.pubNeuroSyM.publish(pred_trajs) 
		#print("pred trajs X0=================== = ",pred_trajs.trajs[0].humans[0].centroid.pose.position.x)
		rospy.loginfo("Finish publishing predicted motion")

		end = rospy.get_rostime()
		#rospy.loginfo("inference runtime %i %i", end.secs-self.start.secs, end.nsecs-self.start.nsecs)
		self.inference_time += ((end.secs-self.start.secs)+ (np.abs(end.nsecs-self.start.nsecs)*(10**(-9))))
		rospy.loginfo("average inference runtime %.9f", out_interface.inference_time/out_interface.batches)
		f = open("inf_time_nonlinear_thor_neurosym.txt", "a")
		f.writelines([str(out_interface.inference_time/out_interface.batches), '\n'])
		f.writelines([str(num_peds_considered),'\n'])
		f.close()
                
		self.data_perception1 = []
		self.message_count_1 = 0
		self.start = rospy.get_rostime()



	def sub2_callback(self, data_percep2):
        

		if self.message_count_2 < self.seq_len:
			#rospy.loginfo("restarting collecting data 2")
			self.data_perception2.append([[data_percep2.header.stamp.nsecs, int(ped.name), ped.position.x, ped.position.y] for ped in data_percep2.people])
			self.message_count_2 += 1
			#rospy.loginfo("nb of messages collected = %d", self.message_count_2)

			if self.message_count_2 == self.seq_len:
				self.process_received_percep2()
        
    
	def process_received_percep2(self,):

		rospy.loginfo("start publishing GT motion")
		future_gt_trajs = HumansTrajs()
		future_gt_trajs.header.stamp =  self.pred_stamp #rospy.Time.now()
		print("gt stamp =", future_gt_trajs.header.stamp)
		future_gt_trajs.trajs = []

		for t in range(self.seq_len):
			future_gt_mot = Humans()
			future_gt_mot.header.stamp = rospy.Time.now()
			data = self.data_perception2[t]
			future_gt_mot.humans = []

			for h in range(len(data)):
				ped = Human()
				ped.id = data[h][1]
				ped.centroid.pose.position.x = data[h][2]
				ped.centroid.pose.position.y = data[h][3]
				future_gt_mot.humans.append(ped)
			future_gt_trajs.trajs.append(future_gt_mot)
		
		self.pubNeuroGT.publish(future_gt_trajs) 
		#print("gt trajs X0=================== = ", future_gt_trajs.trajs[0].humans[0].centroid.pose.position.x)
		        
		rospy.loginfo("Finish publishing GT motion")
		self.data_perception2 = []
		self.data_perception1 = []
		self.message_count_2 = 0


	def get_generator(self, checkpoint):
		args = AttrDict(checkpoint['args'])
		generator = TrajectoryGenerator(
		    obs_len=args.obs_len,
		    pred_len=args.pred_len,
		    embedding_dim=args.embedding_dim,
		    encoder_h_dim=args.encoder_h_dim_g,
		    decoder_h_dim=args.decoder_h_dim_g,
		    mlp_dim=args.mlp_dim,
		    num_layers=args.num_layers,
		    noise_dim=args.noise_dim,
		    noise_type=args.noise_type,
		    noise_mix_type=args.noise_mix_type,
		    pooling_type=args.pooling_type,
		    pool_every_timestep=args.pool_every_timestep,
		    dropout=args.dropout,
		    bottleneck_dim=args.bottleneck_dim,
		    neighborhood_size=args.neighborhood_size,
		    grid_size=args.grid_size,
		    batch_norm=args.batch_norm) # generator creation 
		generator.load_state_dict(checkpoint['g_state']) # we feed the generator , 
		generator.cuda()
		generator.train()
		return generator



	def get_time_complexity(model, inp_size, as_strings=False, print_per_layer_stat=False, verbose = True):
		# Get FLOPS information
		macs, params = get_model_complexity_info(model, inp_size, as_strings, print_per_layer_stat, verbose)

		print("FLOPS:", macs)
		print("Parameters:", params)



if __name__ == '__main__':
    
	print("==================================================Get INPUT FROM DARKO WP2 ==================================================")
	
	args = parser.parse_args()
	out_interface = motpred_pub(args)
	start = rospy.get_rostime().secs

	while not rospy.is_shutdown():

		out_interface.rate.sleep()
		print("passed time ==============", rospy.get_rostime().secs-start)
		#while not rospy.is_shutdown() and out_interface.batches <= 100:


	#out_interface.get_time_complexity(generator, generator_input_size, False,False,True)





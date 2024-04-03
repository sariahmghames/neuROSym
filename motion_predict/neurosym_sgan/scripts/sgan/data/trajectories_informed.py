import logging
import os
import math
import itertools
import numpy as np
import torch
import glob
import json

from torch.utils.data import Dataset
from sgan.data.qtc import qtcc1

logger = logging.getLogger(__name__)


#with open('config.json', 'r') as f:
#    config = json.load(f)

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, obs_traj_weight, pred_traj_weight) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    obs_traj_weight = torch.cat(obs_traj_weight, dim=0).permute(2, 0, 1)
    pred_traj_weight = torch.cat(pred_traj_weight, dim=0).permute(2, 0, 1)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, obs_traj_weight, pred_traj_weight, seq_start_end
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset): # A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t', labels_dir = os.getcwd(), filename= "qtcc1_labels.txt"
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.repeat = 2
        self.labels_dir = labels_dir
        self.filename = filename


        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = [] #  each row is 1 sequence and each sequence is len(seq_len)=nb_frames/seq and in each frame there is nb of peds considered then each x and y 
        seq_list_weight = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                #print("peds in curr seq=", peds_in_curr_seq)
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq_qtc_AB = []
                #curr_seq_weight = np.zeros((math.factorial(len(peds_in_curr_seq))/len(peds_in_curr_seq), 2, self.seq_len)) # weight are copied (same) along dim = 1, 
                
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq): # _ is the index in the list and ped_in is corresponding value
                    #if (idx == 0):
                    #    print("ped_id=", ped_id)
                    #    print("num_ped_considered=", num_peds_considered)
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    # Check if the seq len is right
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx # index of first frame where this ped_id appears
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1 # index of last frame where this ped_id appears
                    #if (idx == 0):
                    #    print("pad front=", pad_front)
                    #    print("pad end=", pad_end)
                    #    print("pad diff=", pad_end-pad_front)
                    if ((pad_end - pad_front != self.seq_len) or (curr_ped_seq.shape[0]!= self.seq_len)):
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) # (2: are x and y) so final shape = 2 x seq len
                    curr_ped_seq = curr_ped_seq # shape : 2 x seq len
                    
                    # Make coordinates relative ("with respect to frames")
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1] # current - prev
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1 # the num of peds that appear in all the frames considered for the seq len (so in all the 16 frames)

                
                ind_excl = [(x*num_peds_considered)+x for x in range(num_peds_considered)]
                idx_qtc_weight = 0
                idx_qtc_weight_prev = 0
                peds_comb_list = list(itertools.product(curr_seq[:num_peds_considered], repeat=self.repeat))
                #for x in itertools.product(curr_seq[:num_peds_considered], curr_seq[1:num_peds_considered]):
                if (self.repeat <= num_peds_considered and num_peds_considered!=1):
                    #curr_seq_weight = np.ones((int(math.factorial(num_peds_considered)/math.factorial(num_peds_considered-self.repeat)), 1, self.seq_len-1)) # weight are copied (same) along dim = 1, 
                    curr_seq_weight = np.ones((num_peds_considered*num_peds_considered, 2, self.seq_len)) # weight are copied (same) along dim = 1, 

                    for tup_idx, tup in enumerate(peds_comb_list):
                        #print("tuple_0=", tup[0])
                        #print("tuple_0_shape=", tup[0].shape)
                        #print("tuple_1=", tup[1])
                        if tup_idx not in ind_excl:
                            #print("num_peds_considered=", num_peds_considered)
                            curr_seq_qtc_AB = qtcc1(tup[0].transpose(), tup[1].transpose(), qbits = 4)
                            labels = [self.labelme(list(x)) for x in curr_seq_qtc_AB]
                            curr_seq_weight[idx_qtc_weight, 0, 1:(self.seq_len)] = labels
                            curr_seq_weight[idx_qtc_weight, 1, 1:(self.seq_len)] = labels
                            curr_seq_weight[idx_qtc_weight, 0, 0] = 0
                            curr_seq_weight[idx_qtc_weight, 1, 0] = 0
                            idx_qtc_weight += 1
                            # if (idx_qtc_weight+idx_qtc_weight_prev == *num_peds_considered):
                            #     idx_qtc_weight_prev = idx_qtc_weight
                            #     idx_qtc_weight += 1
                        else:
                            idx_qtc_weight += 1

                else:
                    curr_seq_weight = np.zeros((1 , 2, self.seq_len))
                    #curr_seq_weight[:, :, 0] = 0

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped # addinf trajectories of peds in 1 seq to another seq
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_weight.append(curr_seq_weight[:(num_peds_considered*num_peds_considered)])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0) # num_sequences, nb peds , seq_len, 2 # num sequences are batched
        seq_list_weight = np.concatenate(seq_list_weight, axis=0) # num_sequences, nb peds , seq_len, 2 # num sequences are batched

        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float) # shape : num_sequences, nb coord , seq_len, 
        self.obs_traj_weight = torch.from_numpy(seq_list_weight[:, :, :self.obs_len]).type(torch.float) # shape : num_sequences, nb coord , seq_len, 

        #print("obs traj shape=", self.obs_traj.shape) # torch.Size([1714, 2, 8])
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.pred_traj_weight = torch.from_numpy(seq_list_weight[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        #print("num peds in seq=", num_peds_in_seq)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist() # np.cumsum of num of peds considered over all sequences
        #print("cum_start_idx=", cum_start_idx) # [0, 3, 6, 9, ... 1712, 1714]
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:]) # end is the nb of peds considered in 1 seq .. [(0,3), (3, 6), (6,9) ....] the first is (0,3) because the first appended sequence has 3 peds considered so 3 trajectories of len = seq_len. so start is index of 1st appended traj in 1 sequence and end is index of last traj considered (or last pedestrian) in that sequence. we are not appending the ped_id considered in this sequence
        ]
        #print("seq_start_end=", self.seq_start_end)

    def __len__(self): # The __len__ function returns the number of samples in our dataset.
        return self.num_seq


    def labelme(self, curr_seq_qtc_AB):
        curr_dir = os.getcwd()
        txt_file =  curr_dir + "/" + self.labels_dir + self.filename
        qtc = []
        with open(txt_file) as f:
            labels = [line.strip() for line in f.readlines()] # removes newline \n character at end of each effective line 
            for x in labels:
                qtc.append(x.split(' ')[1])
            #print("qtcAB=",qtc)
            qtcAB_idx = qtc.index(str(curr_seq_qtc_AB).replace(' ', '')) 
            qtcAB_label = labels[qtcAB_idx].strip()[0]

        return qtcAB_label



    def __getitem__(self, index): # The __getitem__ function loads and returns a sample from the dataset at the given index idx. called internally when we create an instance of the class and pass an index to it
        # called by DataLoader of torch when batching, 

        #if (index ==0):
        #    print("index =", index)
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :], # self.obs_traj[start:end, :] shape is torch.Size([5,2,8]) , torch.Size([5,2,8]), torch.Size([8,2,8]) ...
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :], self.obs_traj_weight[start:end, :], self.pred_traj_weight[start:end, :]
        ]
        #if (index ==0):
        #    print("out shape=", out[0].shape)
        return out

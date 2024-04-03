import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    #print("obj_traj before shape=", torch.cat(obs_seq_list, dim=0).shape) # 81,2,8
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1) # 8, 81, 2
    #print("obj_traj after shape=", obs_traj.shape)
    #print("pred_traj before shape=", torch.cat(pred_seq_list, dim=0).shape)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end
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
        min_ped=1, delim='\t'
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

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = [] #  each row is 1 sequence and each sequence is len(seq_len)=nb_frames/seq and in each frame there is nb of peds considered then each x and y 
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            print("frames =", len(frames))
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            #print("num_sequences =", num_sequences) # 857

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                #print("peds in curr seq=", peds_in_curr_seq)
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
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
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1 # the num of peds that appear in all the frames considered for the seq len (so in all the 16 frames)


                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped # addinf trajectories of peds in 1 seq to another seq
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                #print("reduced num peds=", num_peds_in_seq)
                #break

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0) # num_sequences, nb peds , seq_len, 2 # num sequences are batched
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float) # shape : num_sequences, nb peds , seq_len, (2?) 
        
        print("obs traj shape=", self.obs_traj.shape) # torch.Size([2875, 2, 8]) if shuffle = False
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
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

    def __getitem__(self, index): # The __getitem__ function loads and returns a sample from the dataset at the given index idx. called internally when we create an instance of the class and pass an index to it
        # called by DataLoader of torch when batching, 
        if (index ==0):
            print("index =", index)
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :], # self.obs_traj[start:end, :] shape is torch.Size([5,2,8]) , torch.Size([5,2,8]), torch.Size([8,2,8]) ...
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        if (index ==0):
            print("out shape=", out[0].shape)
        return out

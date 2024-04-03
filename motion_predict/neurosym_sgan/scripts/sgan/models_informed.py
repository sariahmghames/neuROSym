import torch
import torch.nn as nn
import numpy as np


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]): # 2 Linear() if total dims_list size = 3
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        #print("get noise 1 shape=", shape)
        #print("get noise type 1 =", noise_type)
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        #print("get noise 2 shape=", shape)
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""

    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        ) # first attribute is the input size to LSTM encoder

        self.spatial_embedding = nn.Linear(2, embedding_dim) # 2 because the input dim is of dim 2 , embedding_dim is the output shape

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        ) # num_layers for LSTM , return 2 initialisation because (h, c)

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        #print("obs traj shape=", obs_traj.shape) # torch.Size([8, 240, 2]) # not always 240
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2)) # the size -1 is inferred from other dimensions, it will b here : obs_len * batch, tensor.view() return a tensor of different shape
        obs_traj_embedding = obs_traj_embedding.reshape(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch) # hidden state initialisation
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, obs_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep
        self.obs_len = obs_len
        self.mlp_pre_pool_dim_0 = self.obs_len * self.embedding_dim

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet_g(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    obs_len = obs_len
                )

            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim] # h_dim maybe is the output from z layer into decoder, h_dim + bottleneck_dim is input size to z layer 
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)


        self.hidden2pos = nn.Linear(h_dim, 2) # hidden to positions (outputs)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, traj, traj_weight):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = [] # fake (generated by generator) relative (cz is a relative pos output from decoder)
        decoder_input = self.spatial_embedding(last_pos_rel) # first mlp in fig 3 pooling module
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)


        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim)) # -1 can be for batch , next rel pos (predicted) =pos(t+1) - pos(t), 
            curr_pos = rel_pos + last_pos # current pos is pos(t+1) to predict

            if self.pool_every_timestep: # every timestep predicted for the decoder output
                decoder_h = state_tuple[0] # state tuple is hidden state and cell state
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos, traj, traj_weight, self.mlp_pre_pool_dim_0) # pool decoder hidden , as the encoder pooling mechanism 
                decoder_h = torch.cat(
                    [decoder_h.view(-1, self.h_dim), pool_h], dim=1) # initialization of decoder hidden state 
                decoder_h = self.mlp(decoder_h) # for the noise addition 
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1]) # state_tuple[1] is the decoder LSTM cell state

            embedding_input = rel_pos # for next step (t+2) input embedding, should be last rel pos which is rel pos of current prediction 

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0] # state_tuple[0] is the last hidden state of decoder 


class PoolHiddenNet_g(nn.Module): # pooling hidden state of all people present in scene. encoder output and decoder output
    """ Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, obs_len = 8):

        super(PoolHiddenNet_g, self).__init__()

        self.mlp_dim = mlp_dim
        self.obs_len = obs_len
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim


        self.mlp_pre_dim = (self.obs_len * embedding_dim) + h_dim  # 1st mlp in fig 3 (pooling module)
        mlp_pre_pool_dims = [self.mlp_pre_dim, 512, bottleneck_dim] # 2nd mlp in pooling module, mlp_pre_dim is the input size to mlp, mlp hidden layer size = 512, output size = bottleneck_dim

        #self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.spatial_embedding = nn.Linear(2*(self.obs_len), self.embedding_dim*(self.obs_len))


        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)


    def repeat3d(self, tensor, num_reps):
    	"""
    	Inputs:
    	-tensor: 2D tensor of any shape
    	-num_reps: Number of times to repeat each row
    	Outpus:
    	-repeat_tensor: Repeat each row such that: R1, R1, R2, R2
    	"""
    	col_len = tensor.size(1)
    	depth_len = tensor.size(2)
    	tensor = tensor.repeat(1, num_reps, 1) # unsqueeze(dim=1)
    	tensor = tensor.view(-1, col_len, depth_len) # to get the repetition along rows, it was along cols
    	return tensor


    def repeat(self, tensor, num_reps):
    	"""
    	Inputs:
    	-tensor: 2D tensor of any shape
    	-num_reps: Number of times to repeat each row
    	Outpus:
    	-repeat_tensor: Repeat each row such that: R1, R1, R2, R2
    	"""
    	col_len = tensor.size(1)
    	tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
    	tensor = tensor.view(-1, col_len) # to get the repetition along rows, it was along cols
    	return tensor


    #@profile
    def forward(self, h_states, seq_start_end, end_pos, traj, traj_weight, mlp_pre_pool_dim_0):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim) # pooled encoder/decoder hidden state
        """
        pool_h = []
        obs_traj = traj[:self.obs_len, :, :]
        pred_traj_gt = traj[self.obs_len:, :, :]
        obs_traj = obs_traj.permute(1,0,2)
        pred_traj_gt = pred_traj_gt.permute(1,0,2)


        batch_seq_ind = 0
        #print("initial obs traj weight shape =", obs_traj_weight.shape)
        for _, (start, end) in enumerate(seq_start_end): # start and end are indices
            #print("start, end =", (start, end))
            
            #start = start.item()
            #end = end.item()
            num_ped = end - start

            curr_hidden = h_states.view(-1, self.h_dim)[start:end] # in the current considered sequence , torch.Size([2, 32]),
            curr_end_pos = end_pos[start:end]
            curr_pose_history = obs_traj[start:end]

            # Repeat -> H1, H2, "H1, H2"
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1) # 1 is for axis, #  torch.Size([4, 32])
            #print("curr_hidden_1 =", curr_hidden_1.shape)

            # Repeat position -> P1, P2, "P1, P2"
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)   # torch.Size([4, 2])
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)  # torch.Size([4, 2])
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2  # torch.Size([4, 2])

            curr_pose_history_1 = curr_pose_history.repeat(num_ped, 1, 1)
            curr_pose_history_2 = self.repeat3d(curr_pose_history, num_ped) 
            curr_rel_pos_history = curr_pose_history_1 - curr_pose_history_2

            traj_weiit = traj_weight[batch_seq_ind].permute(0,2,1)
            traj_weit = traj_weiit.repeat(1, 1, int(self.embedding_dim/traj_weiit.shape[2]))


            traj_weit = torch.reshape(traj_weit, (traj_weit.size(0), traj_weit.size(1)*traj_weit.size(2)))
            curr_rel_pos_history = torch.reshape(curr_rel_pos_history, (curr_rel_pos_history.size(0), curr_rel_pos_history.size(1)*curr_rel_pos_history.size(2)))
            curr_rel_hist_embedding  = traj_weit.cuda() * self.spatial_embedding(curr_rel_pos_history)

            mlp_h_input = torch.cat([curr_rel_hist_embedding, curr_hidden_1], dim=1)

            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1)
            curr_pool_h = curr_pool_h.max(dim=1)[0] # max pooling the output of 2nd mlp layer of pooling mechanism, max aloing axis =1 which is peds axis 
            pool_h.append(curr_pool_h) # (2,8) # 8 is bottleneck dim (we can try to increase bottleneck dims for full obs_len embedding)
            batch_seq_ind += 1

        pool_h = torch.cat(pool_h, dim=0)

        return pool_h


class PoolHiddenNet_d(nn.Module): # pooling hidden state of all people present in scene. encoder output and decoder output
    """ Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, obs_len=8, pred_len=8
    ):
        super(PoolHiddenNet_d, self).__init__()

        self.mlp_dim = mlp_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        #self.mlp_pre_dim = ((self.obs_len+self.pred_len) * embedding_dim) + h_dim # 1st mlp in fig 3 (pooling module)
        self.mlp_pre_dim =  embedding_dim + h_dim # 1st mlp in fig 3 (pooling module)
        mlp_pre_pool_dims = [self.mlp_pre_dim, 512, bottleneck_dim] # 2nd mlp in pooling module, mlp_pre_dim is the input size to mlp, mlp hidden layer size = 512, output size = bottleneck_dim

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)


    def repeat3d(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        depth_len = tensor.size(2)
        tensor = tensor.repeat(num_reps, 1, 1) # unsqueeze(dim=1)
        tensor = tensor.view(-1, col_len, depth_len) # to get the repetition along rows, it was along cols
        return tensor


    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len) # to get the repetition along rows, it was along cols
        return tensor

    def forward(self, h_states, seq_start_end, end_pos, traj, traj_weight, mlp_pre_pool_dim_0):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        - traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim) # pooled encoder/decoder hidden state
        """
        pool_h = []
        #alpha = []
        obs_traj = traj[:self.obs_len, :, :]
        pred_traj_gt = traj[self.obs_len:, :, :]
        obs_traj = obs_traj.permute(1,0,2)
        traj = traj.permute(1,0,2)
        pred_traj_gt = pred_traj_gt.permute(1,0,2)
        
        #print("initial obs traj weight shape =", obs_traj_weight.shape)
        for _, (start, end) in enumerate(seq_start_end): # start and end are indices
            #print("start, end =", (start, end))
            start = start.item() # for 1 ped and start is ? 
            end = end.item()
            num_ped = end - start

            curr_hidden = h_states.view(-1, self.h_dim)[start:end] # in the current considered sequence , torch.Size([2, 32]),
            curr_end_pos = end_pos[start:end]
            curr_pose_history = traj[start:end]
            #print("curr pose history shape=", curr_pose_history.shape) # 2, 8, 2

            # Repeat -> H1, H2, "H1, H2"
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1) # 1 is for axis, #  torch.Size([4, 32])
            #print("curr_hidden_1 shape=", curr_hidden_1.shape) 

            ## Add qtc informed input to curr_hidden_1


            # Repeat position -> P1, P2, "P1, P2"
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)   # torch.Size([4, 2])
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)  # torch.Size([4, 2])
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2  # torch.Size([4, 2])

            curr_pose_history_1 = curr_pose_history.repeat(num_ped, 1, 1)
            #print("curr pose history_1=", curr_pose_history_1.shape)
            curr_pose_history_2 = self.repeat3d(curr_pose_history, num_ped) 
            #print("curr pose history_2=", curr_pose_history_2.shape)
            curr_rel_pos_history = curr_pose_history_2 - curr_pose_history_1
            #traj_weit = traj_weight[start:end].repeat(1, 1, int(self.embedding_dim/traj_weight.shape[2]))
            #traj_weit = self.repeat3d(traj_weit, num_ped)

            curr_rel_embedding = self.spatial_embedding(curr_rel_pos) # input to 1st mlp layer of pooling mechanism
            #print("mlp_rel_embedding shape =", curr_rel_embedding.shape) # (4,16)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1) # hidden states which is output form encoder/decoder
            #print("mlp_h_input shape =", mlp_h_input.shape) # (4,48) .. (..., 144)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            #print("curr_pool_h shape =", curr_pool_h.shape) # (4,8), 'bottleneck_dim': 8 .. (.., 8)
            #pre_pool = curr_pool_h.view(num_ped, num_ped, -1).max(1) # (2, 8)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0] # max pooling the output of 2nd mlp layer of pooling mechanism, max aloing axis =1 which is peds axis 
            #print("curr_pool_h shape =", curr_pool_h.shape) # (2,8)
            pool_h.append(curr_pool_h) # (2,8)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h



class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    def __init__(
        self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor


    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                    top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,
                                                  curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class TrajectoryGenerator(nn.Module):
    
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = Decoder(
            pred_len, obs_len, 
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )

        if pooling_type == 'pool_net':
            self.pool_net_g = PoolHiddenNet_g(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
                obs_len = obs_len
            )


            self.pool_net_d = PoolHiddenNet_d(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
                obs_len = obs_len,
                pred_len = pred_len
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = (encoder_h_dim) + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed(): # mlp in decoder ? after LSTM ? or before ? should be before 
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            #print("seq_start_end dim =", seq_start_end.size())
            noise_shape = (len(seq_start_end),) + self.noise_dim
            #print("noise shape=", noise_shape)

        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            #print("user_noise is not None")
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        #print("z_decoder shape=", z_decoder.shape)
        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                #start = start.item()
                #end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        #print("z decoder shape=", z_decoder.shape)
        #print("_input shape=", _input.shape)
        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h


    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False


    def forward(self, traj, traj_rel, traj_weight, seq_start_end, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        #print("traj full shape =", traj.shape)
        obs_traj = traj[:self.obs_len, :, :]
        #print("traj obs shape =", obs_traj.shape)
        obs_traj_rel = traj_rel[:self.obs_len, :, :]
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        mlp_pre_pool_dim_0 = self.obs_len * self.embedding_dim
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net_g(final_encoder_h, seq_start_end, end_pos, traj, traj_weight, mlp_pre_pool_dim_0) # pool hidden state of encoder
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input) 
        else:
            noise_input = mlp_decoder_context_input # input to the noise
        
        #print("noise_input dim=", noise_input.shape)
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end, obs_traj, traj_weight)
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        self.embedding_dim = embedding_dim


        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1] # real/fake classifier ..... mlp after encoder so input to mlp is outp of encoder of dim = h_dim, hidden layer of mlp has dim = mlp_dim, output of mlp has dim = 1 
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )# classifier after encoder of discriminator if local, else if global classifier is after pooling which is after encoder of discriminator
        
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim] # input to 2nd mlp layer is h_dim + embedding_dim, mlp hidden layer dim = mlp_dim, 2nd mlp layer outp dim = h_dim
            self.pool_net = PoolHiddenNet_d(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                #mlp_dim=mlp_pool_dims,
                mlp_dim=mlp_dim,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout = dropout,
                obs_len = obs_len,
                pred_len = pred_len
            )


    def forward(self, traj, traj_rel, traj_weight, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel) # output of encoder is hidden state
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion (with neighbors) at the start when combined with
        # trajectory information should help in discriminative behavior.
        
        #print("traj full shape =", traj.shape)
        #print("traj full 0 shape=". traj[0].shape)
        mlp_pre_pool_dim_0 = (self.obs_len+self.pred_len) * self.embedding_dim
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            #print("disc traj shape=", traj.shape)
            #print("disc traj[0] shape=", traj[0].shape)
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0], traj, traj_weight, mlp_pre_pool_dim_0
            )
        scores = self.real_classifier(classifier_input) # feed
        return scores

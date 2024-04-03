# args of original sgan model
loaded model args are : AttrDict({'encoder_h_dim_d': 48, 'neighborhood_size': 2.0, 'pool_every_timestep': False, 'clipping_threshold_g': 2.0, 'delim': 'tab', 'dataset_name': 'hotel', 'print_every': 100, 'skip': 1, 'loader_num_workers': 4, 'd_steps': 1, 'batch_size': 64, 'num_epochs': 200, 'num_layers': 1, 'best_k': 20, 'obs_len': 8, 'pred_len': 8, 'g_steps': 1, 'g_learning_rate': 0.0001, 'l2_loss_weight': 1.0, 'grid_size': 8, 'bottleneck_dim': 8, 'checkpoint_name': 'checkpoint', 'gpu_num': '0', 'restore_from_checkpoint': 1, 'dropout': 0.0, 'noise_mix_type': 'global', 'decoder_h_dim_g': 32, 'pooling_type': 'pool_net', 'use_gpu': 1, 'num_iterations': 8512, 'batch_norm': False, 'noise_type': 'gaussian', 'clipping_threshold_d': 0, 'encoder_h_dim_g': 32, 'checkpoint_every': 300, 'd_learning_rate': 0.001, 'checkpoint_start_from': None, 'timing': 0, 'mlp_dim': 64, 'num_samples_check': 5000, 'd_type': 'global', 'noise_dim': (8,), 'embedding_dim': 16})

# Train neurosym model
python scripts/train_informed2.py --noise_dim 8 --d_type global --pool_every_timestep False  --checkpoint_name checkpoint_cnd --bottleneck_dim 8 --encoder_h_dim_d 48 --batch_size 10 --encoder_h_dim_g 32 --embedding_dim 16 --mlp_dim 64 --decoder_h_dim_g 32 --num_epochs 200 --num_iterations 8512 --noise_mix_type 'global' --noise_type 'gaussian' --labels_dir "scripts/sgan/data/" --filename "cnd_labels.txt" 


# Train causal neurosym model (all dataset thor)
python scripts/train_informed2_causal.py --noise_dim 8 --d_type global --pool_every_timestep 0  --checkpoint_name checkpoint_cnd_causal --bottleneck_dim 8 --encoder_h_dim_d 16 --batch_size 10 --encoder_h_dim_g 8 --embedding_dim 16 --mlp_dim 16 --decoder_h_dim_g 8 --num_epochs 200 --num_iterations 8512 --noise_mix_type 'global' --noise_type 'gaussian' --labels_dir "scripts/sgan/data/" --cndfilename "cnd_labels.txt" --embedding_dim_cvar1 4 --embedding_dim_cvar2 4 --embedding_dim_cvar3 4 --cinf_dir "scripts/datasets/thor_data_all/causal_inference/" --cfilename "causal_matrix_average.csv" --delim ',' --div_data 1 --dataset_name thor_data_all/thor_data --pred_len 8 --obs_len 8 --g_learning_rate 0.0001 --d_learning_rate 0.0001


# Train on subset thor - causal
python scripts/train_informed2_causal.py --noise_dim 8 --d_type global --pool_every_timestep 0  --checkpoint_name checkpoint_cnd_causal --bottleneck_dim 8 --encoder_h_dim_d 16 --batch_size 10 --encoder_h_dim_g 8 --embedding_dim 16 --mlp_dim 16 --decoder_h_dim_g 8 --num_epochs 200 --num_iterations 8500 --noise_mix_type 'global' --noise_type 'gaussian' --labels_dir "scripts/sgan/data/" --cndfilename "cnd_labels.txt" --embedding_dim_cvar1 4 --embedding_dim_cvar2 4 --embedding_dim_cvar3 4 --cinf_dir "scripts/datasets/thor_data_all/causal_inference/" --cfilename "causal_matrix_average.csv" --delim ',' --div_data 1 --dataset_name thor_data_all/thor_data --pred_len 8 --obs_len 8 --g_learning_rate 0.0001 --d_learning_rate 0.0001



- For training with automatically generated cnd labels, pass cnd_labels.txt to arg filename
- For training with expert guessed binary labels, use qtcc1_labels.txt as value of arg filename

# Evaluation
python scripts/evaluate_model.py --model_path ./scripts/models/new/noalpha_8ts_zara1/checkpoint_noalpha_zara1_with_model.pt --save 1
python scripts/evaluate_model_informed_causal.py --model_path ./scripts/models/new/thor_full_dataset/checkpoint_cnd_causal_with_model.pt --save 1 --dset_type test


# Plot motion
python plot_motion_full.py --traj_path_pred_gt ./models/new/ARENA/visualise_ARENA_gt_pred_test/ARENA_full_test_pred_gt.pt --traj_path_pred_est ./models/new/ARENA/visualise_ARENA_gt_pred_test/ARENA_full_test_pred_est.pt --traj_path_obs ./models/new/ARENA/visualise_ARENA_gt_pred_test/ARENA_full_test_obs_gt.pt



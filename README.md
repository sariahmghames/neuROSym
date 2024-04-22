<p align="center">
    <img src="https://github.com/sariahmghames/neuROSym/blob/main/logo.png" width="285" height="100" /> 
</p> 
A ROS package for online human motion prediction and visualization 

This repository contains mainly 3 ROS nodes:

* Inference node
* Visualisation node
* Bags post-processing node

We note that the training scripts used to generate the models for the inference phase can be found @ https://github.com/sariahmghames/NeuroSyM-prediction

Create a ROS workspace as follows:
```
mkdir -p ~/neuROSym/src
cd ~/neuROSym/src
catkin_make
source ~/neuROSym/devel/setup.bash
```

Currently, the inference and visualisation nodes are runned in separate terminals, after running the following in 2 separate terminals:
```
cd ~/neuROSym/src/motion_predict/neurosym_sgan/scripts
```

Then run separately the following:

```
python3 inference_script_name --model_path model_path_name
```

```
python3 plot_inference.py 
```

Replace the inference_script_name with any of the following:

* inference_model_informed_velodyne_pedfiltered.py
* inference_model_velodyne_pedfiltered.py

and replace the model_path_name with any of the following:

* ~/neuROSym/src/motion_predict/neurosym_sgan/models/thor_full_dataset/checkpoint_alpha_cnd_nocausal_opt_8ts_thor_with_model.pt
* ~/neuROSym/src/motion_predict/neurosym_sgan/models/thor_full_dataset/checkpoint_nocnd_nocausal_8ts_thor_with_model.pt
* ~/neuROSym/src/motion_predict/neurosym_sgan/models/zara1/checkpoint_alpha_cnd_nocausal_opt_8ts_zara1_with_model.pt
* ~/neuROSym/src/motion_predict/neurosym_sgan/models/zara1/checkpoint_nocnd_nocausal_8ts_zara1_with_model.pt


```
We welcome any issue or collaborations. You can reach out @ sariahmghames@gmail.com and/or lucacastri94@gmail.com
```

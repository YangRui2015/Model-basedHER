# Offline MHER
Our code is based on WGCSL, whose original anonymous code can be found here [https://github.com/YangRui2015/AWGCSL](https://github.com/YangRui2015/AWGCSL).


## Requirements
python3.6+, tensorflow, gym, mujoco, mpi4py

## Installation
- Clone the repo and cd into it:

- Install baselines package
    ```bash
    pip install -e .
    ```


## Usage
Environments: Point2DLargeEnv-v1, Point2D-FourRoom-v1, FetchReach-v1, SawyerReachXYZEnv-v1, Reacher-v2, SawyerDoor-v0.

offline MHER (note: you can set alpha as the parameter used in our paper):
```bash
python3 -m  wgcsl.run  --env=FetchReach-v1 --num_env 1 --mode model --log_path ~/${path_name}  --alpha 5  --offline_train --load_buffer --load path  ./offline_data/expert/FetchReach-v1/
```


Offline WGCSL
```bash
python3 -m  wgcsl.run  --env=FetchReach-v1 --num_env 1 --mode supervised  --random_init 0 --load_path ./offline_data/random/FetchReach-v1/ --load_buffer --su_method gamma_exp_adv_clip10  --offline_train 
```

Offline GCSL
```bash
python3 -m  wgcsl.run  --env=FetchReach-v1 --num_env 1 --mode supervised  --random_init 0 --load_path ./offline_data/random/FetchReach-v1/ --load_buffer  --offline_train 
```

Goal MARVIL
```bash
python3 -m  wgcsl.run  --env=FetchReach-v1 --num_env 1 --mode supervised  --random_init 0 --load_path ./offline_data/random/FetchReach-v1/ --load_buffer  --su_method exp_adv  --no_relabel True   --offline_train 
```

Goal Behavior Cloning
```bash
python3 -m  wgcsl.run  --env=FetchReach-v1 --num_env 1 --mode supervised  --random_init 0 --load_path ./offline_data/random/FetchReach-v1/ --load_buffer     --offline_train 
```

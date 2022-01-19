# Model-based Hindsight Experience Replay(MHER)
Code for Model-based Hindisight Experience Replay (MHER) . MHER is a novel algorithm leveraging model-based achieved goals for both goal relabeling and policy improvement.

<div style="text-align: center;">
<img src="pics/diagram.png"  >
</div>


## Requirements
python3.6+, tensorflow, gym, mujoco, mpi4py

## Installation
- Clone the repo and cd into it:

- Install baselines package
    ```bash
    pip install -e .
    ```


## Usage
Environments: Point2DLargeEnv-v1, Point2D-FourRoom-v1, FetchReach-v1, SawyerReachXYZEnv-v1, Reacher-v2.

MHER:
```bash
python -m  mher.run --env=Point2DLargeEnv-v1 --num_epoch 30 --num_env 1  --n_step 5 --mode dynamic --alpha 3 --mb_relabeling_ratio 0.8 --log_path=~/logs/point/ --save_path=~/logs/point/
```
MHER without MGSL (DDPG + MBR)
```bash
python -m  mher.run --env=Point2DLargeEnv-v1 --num_epoch 30 --num_env 1  --n_step 5 --mode dynamic --alpha 0 --mb_relabeling_ratio 0.8  --no_mgsl True 
```
MHER without MBR (DDPG + SL)
```bash
python -m  mher.run --env=Point2DLargeEnv-v1 --num_epoch 30 --num_env 1  --n_step 5 --mode dynamic --mb_relabeling_ratio 0.8  --no_mb_relabel True
```

DDPG:
```bash
python -m  mher.run  --env=Point2DLargeEnv-v1 --num_epoch 30 --num_env 1 --noher True 
```
HER:
```bash
python -m  mher.run  --env=Point2DLargeEnv-v1 --num_epoch 30 --num_env 1 
```
GCSL:
```bash
python -m  mher.run  --env=Point2DLargeEnv-v1 --num_epoch 30 --num_env 1 --mode supervised
```

Model-based Policy Optimization(MBPO):
```bash
python -m mher.run  --env=FetchReach-v1  --num_epoch 30 --num_env 1 --mode mbpo  --n_step 5
```

Model-based Value Expansion(MVE):
```bash
python -m mher.run  --env=FetchReach-v1  --num_epoch 30 --num_env 1 --mode mbpo  --n_step 5
```

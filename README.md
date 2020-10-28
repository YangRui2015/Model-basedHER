# baselines_new
baselines_new supports multi-step HER and is revised from OpenAI baselines.

## Prerequisites 
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu 
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
    
### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

## Tensorflow versions
The master branch supports Tensorflow from version 1.4 to 1.14. For Tensorflow 2.0 support, please use tf2 branch.

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines

- Install baselines package
    ```bash
    pip install -e .
    ```


## Usage

## Usage
DDPG:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12 --noher True --log_path=~/logs/FetchPush_env12/ --save_path=~/ddpg/fetchpush/
```
$\lambda$ n-step DDPG:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12 --mode lambda --lamb 0.7 --n_step 2 --noher True --log_path=~/logs/FetchPush_env12/ --save_path=~/lddpg/fetchpush/
```
Model-based n-step DDPG:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12 --mode dynamic --alpha 0.5 --n_step 2 --noher True --log_path=~/logs/FetchPush_env12/ --save_path=~/mddpg/fetchpush/
```
HER:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12 --log_path=~/logs/FetchPush_env12/ --save_path=~/her/fetchpush/
```
HER + Multi-step:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12  --n_step 2 --mode nstep --log_path=~/logs/FetchPush_env12_nstep_2/ --save_path=~/policies/nstepher/fetchpush/
```
HER + Correction:
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12  --n_step 2 --mode correct --cor_rate 1 --log_path=~/logs/FetchPush_env12_nstep_2/ --save_path=~/policies/nstepher/fetchpush/
```
NHER($\lambda$):
```bash
python -m  baselines.run  --env=FetchPush-v1 --num_epoch 50 --num_env 12  --n_step 2 --mode lambda --lamb 0.7 --log_path=~/logs/FetchPush_env12_nstep_2/ --save_path=~/policies/nher_lambda/fetchpush/
```
Model-based NHER:
```bash
python -m  baselines.run --env=FetchPush-v1 --num_epoch 50 --num_env 12  --n_step 2 --mode dynamic --alpha 0.5 --log_path=~/logs/FetchPush_env12_nstep_2/ --save_path=~/policies/mnher/fetchpush/
```


## Update
* 6.11 first update of n-step her, add support of num_epoch;
* 7.02 update action threshold method for correction;
* 7.12 update taylor correction;
* 8.2 update lambda multi-step HER
* 8.23 update model-based multi-step HER
* 10.28 merge old code and new code, update readme


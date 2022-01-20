import numpy as np
import os
import gym
from wgcsl.envs import register_envs
from gym.wrappers import FlattenObservation
from wgcsl.common.monitor import Monitor
from wgcsl.common.wrappers import ClipActionsWrapper
register_envs()


def make_env(env_id,  mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    if ':' in env_id:
        import re
        import importlib
        module_name = re.sub(':.*','',env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)
    env = gym.make(env_id, **env_kwargs)

    if env_id.startswith('Fetch'):
        from wgcsl.envs.multi_world_wrapper import FetchGoalWrapper
        env._max_episode_steps = 50
        env = FetchGoalWrapper(env)
    elif env_id.startswith('Hand'):
        env._max_episode_steps = 100
    elif env_id.startswith('Sawyer'):
        from wgcsl.envs.multi_world_wrapper import SawyerGoalWrapper
        env = SawyerGoalWrapper(env)
        if not hasattr(env, '_max_episode_steps'):
            env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    elif env_id.startswith('Point'):
        from wgcsl.envs.multi_world_wrapper import PointGoalWrapper
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
        env = PointGoalWrapper(env)
    elif env_id.startswith('Reacher'):
        from wgcsl.envs.multi_world_wrapper import ReacherGoalWrapper
        env._max_episode_steps = 50
        env = ReacherGoalWrapper(env)
    else:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
        # raise NotImplementedError('No such environment till now.')

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)
    return env

dirs = ['expert', 'random']
for path in dirs:
    path = './' + path + '/'
    new_ds = os.listdir(path)
    print(path)
    for ds in new_ds:
        if ds == '.DS_Store':
            continue
        pkl_path = os.path.join(path, ds, 'buffer.pkl')
        data = np.load(pkl_path, allow_pickle=True)
        ag2 = data['ag'][:, 1:, :]
        g = data['g']
        env = make_env(ds)
        if 'Point' in ds:
            try:
                env.env.env.env.env.env.env.reward_type = 'sparse'
            except:
                pass
        rewards = np.zeros((g.shape[0:2]))
        for i in range(g.shape[0]):
            try:
                rewards[i] = env.compute_rewards(ag2[i], g[i], None)
            except:
                rewards[i] = env.compute_reward(ag2[i], g[i], None)
        success_rate = 0
        if rewards.max() == 0 and rewards.min() == -1:
            success_rate = (rewards[:, -1] == 0)
        elif rewards.max() == 1 and rewards.min() == 0:
            success_rate = (rewards[:, -1] == 1)
        else:
            import pdb;pdb.set_trace()
        print(ds + ' mean: {}, std: {}'.format( success_rate.mean(), success_rate.std()))


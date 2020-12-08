from tensorflow.python.framework.ops import IndexedSlices
from baselines.her.util import transitions_in_episode_batch
import numpy as np
import gym
import multiworld
from numpy.core.defchararray import index
from numpy.lib.index_tricks import AxisConcatenator

def make_random_sample(reward_fun):
    def _random_sample(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # # Re-compute reward since we may have substituted the u and o_2 ag_2
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions
    return _random_sample
        
def obs_to_goal_fun(env):
    # only support Fetchenv and Handenv now
    from gym.envs.robotics import FetchEnv, hand_env
    from multiworld.envs.pygame import point2d
    from multiworld.envs.mujoco.sawyer_xyz import sawyer_push_and_reach_env
    from multiworld.envs.mujoco.sawyer_xyz import sawyer_reach
    from gym.envs.mujoco import reacher

    if isinstance(env.env, FetchEnv):
        obs_dim = env.observation_space['observation'].shape[0]
        goal_dim = env.observation_space['desired_goal'].shape[0]
        temp_dim = env.sim.data.get_site_xpos('robot0:grip').shape[0]
        def obs_to_goal(observation):
            observation = observation.reshape(-1, obs_dim)
            if env.has_object:
                goal = observation[:, temp_dim:temp_dim + goal_dim]
            else:
                goal = observation[:, :goal_dim]
            return goal.copy()
    elif isinstance(env.env, hand_env.HandEnv):
        goal_dim = env.observation_space['desired_goal'].shape[0]
        def obs_to_goal(observation):
            goal = observation[:, -goal_dim:]
            return goal.copy()
    elif isinstance(env.env, point2d.Point2DEnv):
        def obs_to_goal(observation):
            return observation.copy()
    elif isinstance(env.env, sawyer_push_and_reach_env.SawyerPushAndReachXYZEnv):
        assert env.env.observation_space['observation'].shape == env.env.observation_space['achieved_goal'].shape, \
            "This environment's observation space doesn't equal goal space"
        def obs_to_goal(observation):
            return observation
    elif isinstance(env.env, sawyer_reach.SawyerReachXYZEnv):
        def obs_to_goal(observation):
            return observation
    elif isinstance(env.env.env, reacher.ReacherEnv):
        def obs_to_goal(observation):
            return observation[:, -3:-1]
    else:
        def obs_to_goal(observation):
            return observation
        # raise NotImplementedError('Do not support such type {}'.format(env))
        
    return obs_to_goal


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, obs_to_goal_fun=None, no_her=False):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    if no_her:
        print( '*' * 10 + 'Do not use HER in this method' + '*' * 10)
    
    def _random_log(string):
        if np.random.random() < 0.002:
            print(string)
    
    def _preprocess(episode_batch, batch_size_in_transitions):
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout

        # Select which episodes and time steps to use. 
        # np.random.randint doesn't contain the last one, so comes from 0 to roolout_batch_size-1
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        return transitions, episode_idxs, t_samples, batch_size, T

    def _get_reward(ag_2, g):
        # Reconstruct info dictionary for reward  computation.
        info = {}
        # for key, value in transitions.items():
        #     if key.startswith('info_'):
        #         info[key.replace('info_', '')] = value
        # Re-compute reward since we may have substituted the goal.
        reward_params = {'ag_2':ag_2, 'g':g}
        reward_params['info'] = info
        return reward_fun(**reward_params)

    def _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T):
        her_indexes = (np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)  #np.minimum(T - t_samples, 3)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        return future_ag.copy(), her_indexes.copy()

    def _get_dynamic_ags(transitions, batch_size, action_fun, model, ratio=0.3, steps=2):
        indexs = (np.random.uniform(size=batch_size) < ratio)
        states, goals = transitions['o'][indexs], transitions['g'][indexs]
        next_states = states
        for i in range(steps):
            actions = action_fun(states, goals)
            next_states = model.predict_next_state(states, actions)
            states = next_states

        next_goals = obs_to_goal_fun(next_states)
        return next_goals.copy(), indexs.copy()
    
    def _get_ags_from_states(batch_size, states, ratio=0.3):
        indexs = (np.random.uniform(size=batch_size) < ratio)
        next_goals = obs_to_goal_fun(states[indexs])
        return next_goals.copy(), indexs.copy()
        
    def _reshape_transitions(transitions, batch_size, batch_size_in_transitions):
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions


    def _sample_her_transitions(episode_batch, batch_size_in_transitions, info=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        
        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            transitions['g'][her_indexes] = future_ag

        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)


    def _sample_nstep_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, Q_fun = info['nstep'], info['gamma'], info['get_Q_pi']
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        _random_log('using nstep sampler with step:{}'.format(steps))

        assert steps < T, 'Steps should be much less than T.'

        n_step_ags = np.zeros((batch_size, steps, episode_batch['ag'].shape[-1]))
        n_step_reward_mask = np.ones((batch_size, steps)) * np.array([pow(gamma,i) for i in range(steps)])
        for i in range(steps):
            i_t_samples = t_samples + i
            n_step_reward_mask[:,i][np.where(i_t_samples > T - 1)] = 0
            i_t_samples[i_t_samples > T-1] = T-1   # last state to compute reward
            n_step_ags[:,i,:] = episode_batch['ag_2'][episode_idxs, i_t_samples]

        i_t_samples = t_samples + steps # last state to observe
        i_t_samples[i_t_samples > T] = T
        n_step_os = episode_batch['o'][episode_idxs, i_t_samples]
        n_step_gamma = np.ones((batch_size,1)) * pow(gamma, steps)

        # use inverse order to find the first zero in each row of reward mask
        for i in range(steps-1, 0, -1):
            n_step_gamma[np.where(n_step_reward_mask[:,i] == 0)] = pow(gamma, i)

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            transitions['g'][her_indexes] = future_ag
        
        n_step_gs = transitions['g'].repeat(steps, axis=0)
        # Re-compute reward since we may have substituted the goal.
        ags = n_step_ags.reshape((batch_size * steps, -1))
        n_step_reward = _get_reward(ags, n_step_gs)

        transitions['r'] = (n_step_reward.reshape((batch_size, steps)) * n_step_reward_mask).sum(axis=1).copy()
        transitions['r'] += (n_step_gamma * Q_fun(o=n_step_os, g=transitions['g'])).reshape(-1)
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)
         
    
    def _sample_nstep_correct_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, cor_rate = info['nstep'], info['gamma'], info['cor_rate']
        Q_pi_fun, Q_fun = info['get_Q_pi'], info['get_Q']
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        assert steps < T, 'Steps should be much less than T.'

        _random_log('using nstep correct sampler with step:{} and cor_rate:{}'.format(steps, cor_rate))

        n_step_ags = np.zeros((batch_size, steps, episode_batch['ag'].shape[-1]))
        n_step_reward_mask = np.ones((batch_size, steps)) * np.array([pow(gamma,i) for i in range(steps)])
        n_step_o2s= np.zeros((batch_size, steps, episode_batch['o'].shape[-1]))
        n_step_us = np.zeros((batch_size, steps, episode_batch['u'].shape[-1]))
        n_step_gamma_matrix = np.ones((batch_size, steps))  # for lambda * Q

        for i in range(steps):
            i_t_samples = t_samples + i
            n_step_reward_mask[:,i][np.where(i_t_samples > T - 1)] = 0
            n_step_gamma_matrix[:,i] = pow(gamma, i+1)
            if i >= 1:  # more than length, use the last one
                n_step_gamma_matrix[:,i][np.where(i_t_samples > T -1)] = n_step_gamma_matrix[:, i-1][np.where(i_t_samples > T-1)]
            i_t_samples[i_t_samples > T-1] = T-1   # last state to compute reward
            n_step_ags[:,i,:] = episode_batch['ag_2'][episode_idxs, i_t_samples]
            n_step_o2s[:,i, :] = episode_batch['o_2'][episode_idxs, i_t_samples]
            n_step_us[:,i,:] = episode_batch['u'][episode_idxs, i_t_samples]

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            transitions['g'][her_indexes] = future_ag

        n_step_gs = transitions['g'].repeat(steps, axis=0)
        # Reconstruct info dictionary for reward  computation.
        ags = n_step_ags.reshape((batch_size * steps, -1))
        n_step_reward = _get_reward(ags, n_step_gs)

        transitions['r'] = (n_step_reward.reshape((batch_size, steps)) * n_step_reward_mask).sum(axis=1).copy()
        transitions['o_2'] = n_step_o2s[:, -1, :].reshape((batch_size, episode_batch['o'].shape[-1])).copy()
        transitions['gamma'] = n_step_gamma_matrix[:, -1].copy()
        transitions['r'] += transitions['gamma'] * Q_pi_fun(o=transitions['o_2'], g=transitions['g']).reshape(-1)

        correction = 0
        for i in range(steps - 1):
            obs = n_step_o2s[:, i, :].reshape((batch_size, episode_batch['o'].shape[-1]))
            acts = n_step_us[:, i+1,:].reshape((batch_size, episode_batch['u'].shape[-1]))
            correction += n_step_reward_mask[:, i+1] * (Q_pi_fun(o=obs, g=transitions['g'].reshape(-1)) - Q_fun(o=obs, g=transitions['g'].reshape(-1),u=acts)).reshape(-1) 
        transitions['r']  += correction * cor_rate
        # if np.random.random() < 0.1:
        #     from baselines.her.util import write_to_file
        #     write_to_file(str(correction.mean()))

        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    def _lambda_nstep_process(episode_batch, transitions, episode_idxs, t_samples, batch_size, T, steps, gamma, lamb, Q_fun):
        n_step_ags = np.zeros((batch_size, steps, episode_batch['ag'].shape[-1]))
        n_step_reward_mask = np.ones((batch_size, steps)) * np.array([pow(gamma,i) for i in range(steps)])
        n_step_o2s= np.zeros((batch_size, steps, episode_batch['o'].shape[-1]))
        n_step_gamma_matrix = np.ones((batch_size, steps))  # for lambda * Q

        for i in range(steps):
            i_t_samples = t_samples + i
            n_step_reward_mask[:,i][np.where(i_t_samples > T - 1)] = 0
            n_step_gamma_matrix[:,i] = pow(gamma, i+1)
            if i >= 1:  # more than length, use the last one
                n_step_gamma_matrix[:,i][np.where(i_t_samples > T -1)] = n_step_gamma_matrix[:, i-1][np.where(i_t_samples > T-1)]
            i_t_samples[i_t_samples > T-1] = T-1   # last state to compute reward
            n_step_ags[:,i,:] = episode_batch['ag_2'][episode_idxs, i_t_samples]
            n_step_o2s[:,i,:] = episode_batch['o_2'][episode_idxs, i_t_samples]

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        n_step_gs = transitions['g'].repeat(steps, axis=0)
        ags = n_step_ags.reshape((batch_size * steps, -1))
        n_step_reward = _get_reward(ags, n_step_gs)

        return_array = np.zeros((batch_size, steps))
        return_mask = np.ones((batch_size, steps)) * np.array([pow(lamb,i) for i in range(steps)])
        # return_mask[n_step_reward_mask == 0] = 0
        for i in range(steps):
            return_i = (n_step_reward.reshape((batch_size, steps))[:,:i+1] * n_step_reward_mask[:,:i+1]).sum(axis=1) + n_step_gamma_matrix[:, i] * Q_fun(
                        o=n_step_o2s[:,i,:].reshape((batch_size, episode_batch['o'].shape[-1])), 
                        g=transitions['g']).reshape(-1)
            return_array[:, i] = return_i.copy()
        lambda_return = ((return_array * return_mask).sum(axis=1) / return_mask.sum(axis=1)).copy()
        return lambda_return
    
    def _sample_nstep_lambda_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, Q_fun, lamb = info['nstep'], info['gamma'], info['get_Q_pi'], info['lamb']
        transitions, episode_idxs, t_samples, batch_size, T= _preprocess(episode_batch, batch_size_in_transitions)
        assert steps < T, 'Steps should be much less than T.'

        _random_log('using nstep lambda sampler with step:{} and lamb:{}'.format(steps, lamb))

        transitions['r'] = _lambda_nstep_process(episode_batch, transitions, episode_idxs, t_samples, batch_size, T, steps, gamma, lamb, Q_fun)
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)


    def _sample_nstep_dynamic_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, Q_fun, alpha = info['nstep'], info['gamma'], info['get_Q_pi'], info['alpha']
        dynamic_model, action_fun = info['dynamic_model'], info['action_fun']
        dynamic_ag_ratio = info['mb_relabeling_ratio']
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        train_policy = info['train_policy']
        get_rate = info['get_rate']
        process_rate = get_rate()

        min_rate = 0.1
        dynamic_ag_ratio_cur = max(dynamic_ag_ratio - min_rate, 0) * (1 - process_rate) + min_rate if dynamic_ag_ratio > 0 else 0

        _random_log('using nstep dynamic sampler with step:{}, alpha:{}, and dynamic relabeling rate:{}'.format(steps, alpha, dynamic_ag_ratio_cur))
        # preupdate dynamic model
        loss = dynamic_model.update(transitions['o'], transitions['u'], transitions['o_2'], times=2)  

        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            transitions['g'][her_indexes] = future_ag

        # # Re-compute reward since we may have substituted the goal.
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])

        ## model-based on-policy, when 1 step then is exactly  
        reward_list = [transitions['r']]
        last_state = transitions['o_2']
        if steps > 1 and dynamic_ag_ratio_cur > 0:
            next_states_list = []
            for _ in range(1, steps):
                state_array = last_state
                action_array = action_fun(o=state_array, g=transitions['g'])
                next_state_array = dynamic_model.predict_next_state(state_array, action_array)
                
                if np.random.random() < 0.1:
                    # test loss
                    predicted_obs = dynamic_model.predict_next_state(transitions['o'], transitions['u'])
                    loss = np.abs((transitions['o_2'] - predicted_obs)).mean() 
                    print(loss)

                next_states_list.append(next_state_array.copy())
                last_state = next_state_array

            # # # add dynamic achieve goals
            new_ags, indexes = _get_ags_from_states(batch_size, last_state, dynamic_ag_ratio_cur)
            transitions['g'][indexes] = new_ags
            train_policy(o=transitions['o'][indexes], g=new_ags, u=transitions['u'][indexes])   #transitions['g'][indexes]
                
            # recompute rewards
            reward_list[0] = _get_reward(transitions['ag_2'], transitions['g'])

        target_step1 = reward_list[0] + gamma * Q_fun(o=transitions['o_2'], g=transitions['g']).reshape(-1)
        transitions['r'] = target_step1.copy()
        
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    return _sample_her_transitions, _sample_nstep_her_transitions, _sample_nstep_correct_her_transitions, \
             _sample_nstep_lambda_her_transitions , _sample_nstep_dynamic_her_transitions




    # def _sample_nstep_dynamic_her_transitions(episode_batch, batch_size_in_transitions, info):
    #     steps, gamma, Q_fun, alpha = info['nstep'], info['gamma'], info['get_Q_pi'], info['alpha']
    #     dynamic_model, action_fun = info['dynamic_model'], info['action_fun']
    #     dynamic_ag_ratio = info['mb_relabeling_ratio']
    #     transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)

    #     _random_log('using nstep dynamic sampler with step:{}, alpha:{}, and dynamic relabeling rate:{}'.format(steps, alpha, dynamic_ag_ratio))

    #     # preupdate dynamic model
    #     loss = dynamic_model.update(transitions['o'], transitions['u'], transitions['o_2'], times=2)  
    #     # if np.random.random() < 0.1:
    #     #     print(loss)

    #     # Select future time indexes proportional with probability future_p. These
    #     # will be used for HER replay by substituting in future goals.
    #     if not no_her:
    #         future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
    #         transitions['g'][her_indexes] = future_ag

    #     # Re-compute reward since we may have substituted the goal.
    #     transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])

    #     ## model-based on-policy, when 1 step then is exactly  
    #     reward_list = [transitions['r']]
    #     last_state = transitions['o_2']

    #     if steps > 1:
    #         next_states_list = []
    #         for _ in range(1, steps):
    #             state_array = last_state
    #             action_array = action_fun(o=state_array, g=transitions['g'])
    #             next_state_array = dynamic_model.predict_next_state(state_array, action_array)
                
    #             if np.random.random() < 0.1:
    #                 # test loss
    #                 predicted_obs = dynamic_model.predict_next_state(transitions['o'], transitions['u'])
    #                 loss = np.abs((transitions['o_2'] - predicted_obs)).mean() 
    #                 print(loss)

    #             next_states_list.append(next_state_array.copy())
    #             last_state = next_state_array

    #         # # # add dynamic achieve goals
    #         # if dynamic_ag_ratio >= 0:
                
    #         #     new_ags, indexes = _get_ags_from_states(batch_size, last_state, dynamic_ag_ratio)
    #         #     transitions['g'][indexes] = new_ags
                
    #             # clip ags do not work on 
    #             # real_ags = obs_to_goal_fun(state_array[her_indexes])
    #             # less_idx, more_idx = np.where(new_ags < real_ags), np.where(new_ags >= real_ags)
    #             # new_ags[less_idx] = np.clip(new_ags[less_idx], real_ags[less_idx] - 0.2 * np.abs(real_ags[less_idx]), real_ags[less_idx] - 0.02 * np.abs(real_ags[less_idx]) )
    #             # new_ags[more_idx] = np.clip(new_ags[more_idx], real_ags[more_idx] + 0.02 * np.abs(real_ags[more_idx]),  + real_ags[more_idx] + 0.2 * np.abs(real_ags[more_idx]))
    #             # FetchReach 0.02, 
    #             # new_ags = np.clip(new_ags, real_ags - 0.2 * np.abs(real_ags), real_ags + 0.2 * np.abs(real_ags))

    #             # if np.random.random()  < 0.05:
    #             #     relative_goal_dis = ((new_ags - real_ags) / real_ags).mean(axis=0)
    #             #     print(relative_goal_dis)
    #             # print(((new_ags - real_ags) / np.abs(real_ags)).mean(axis=0), ((future_ag - real_ags)/ np.abs(real_ags)).mean(axis=0))
    #             # print(_get_reward(transitions['ag_2'], transitions['g']).mean())
                
    #             # new_reward = _get_reward(new_ags, transitions['ag_2'][indexes])
    #             # print(new_reward.mean())
                
    #         # recompute rewards
    #         reward_list[0] = _get_reward(transitions['ag_2'], transitions['g'])
    #         for j in range(steps - 1):
    #             next_reward = _get_reward(obs_to_goal_fun(next_states_list[j]), transitions['g'])
    #             reward_list.append(next_reward.copy())

    #     last_Q = Q_fun(o=last_state, g=transitions['g'])
    #     target = 0
    #     for i in range(steps):
    #         target += pow(gamma, i) * reward_list[i]
    #     target += pow(gamma, steps) * last_Q.reshape(-1)
    #     transitions['r'] = target.copy()
    #     # allievate the model bias
    #     if steps > 1:
    #         target_step1 = reward_list[0] + gamma * Q_fun(o=transitions['o_2'], g=transitions['g']).reshape(-1)
    #         transitions['r'] = (alpha * transitions['r'] + target_step1) / (1 + alpha)
        
    #     return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)
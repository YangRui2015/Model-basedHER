import numpy as np
import gym
import multiworld

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
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        return future_ag.copy(), her_indexes.copy()

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

    
    def _sample_nstep_lambda_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, Q_fun, lamb = info['nstep'], info['gamma'], info['get_Q_pi'], info['lamb']
        transitions, episode_idxs, t_samples, batch_size, T= _preprocess(episode_batch, batch_size_in_transitions)
        assert steps < T, 'Steps should be much less than T.'

        _random_log('using nstep lambda sampler with step:{} and lamb:{}'.format(steps, lamb))

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
        transitions['r'] = ((return_array * return_mask).sum(axis=1) / return_mask.sum(axis=1)).copy()
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)


    def _sample_nstep_dynamic_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, gamma, Q_fun, alpha = info['nstep'], info['gamma'], info['get_Q_pi'], info['alpha']
        dynamic_model, action_fun = info['dynamic_model'], info['action_fun']
        transitions, episode_idxs, t_samples, batch_size, T= _preprocess(episode_batch, batch_size_in_transitions)

        _random_log('using nstep dynamic sampler with step:{} and alpha:{}'.format(steps, alpha))

        # preupdate dynamic model
        loss = dynamic_model.update(transitions['o'], transitions['u'], transitions['o_2'], times=2)
        # print(loss)

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])

        ## model-based on-policy
        reward_list = [transitions['r']]
        last_state = transitions['o_2']
        if steps > 1:
            for _ in range(1, steps):
                state_array = last_state
                action_array = action_fun(o=state_array, g=transitions['g'])
                next_state_array = dynamic_model.predict_next_state(state_array, action_array)
                # test loss
                predicted_obs = dynamic_model.predict_next_state(state_array, transitions['u'])
                loss = np.abs((transitions['o_2'] - predicted_obs)).mean()
                if np.random.random() < 0.1:
                    print(loss)
                    # print(transitions['o_2'][0])
                    # print(predicted_obs[0])

               
                next_reward = _get_reward(obs_to_goal_fun(next_state_array), transitions['g'])
                reward_list.append(next_reward.copy())
                last_state = next_state_array

        last_Q = Q_fun(o=last_state, g=transitions['g'])
        target = 0
        for i in range(steps):
            target += pow(gamma, i) * reward_list[i]
        target += pow(gamma, steps) * last_Q.reshape(-1)
        transitions['r'] = target.copy()
        # allievate the model bias
        if steps > 1:
            target_step1 = reward_list[0] + gamma * Q_fun(o=transitions['o_2'], g=transitions['g']).reshape(-1)
            transitions['r'] = (alpha * transitions['r'] + target_step1) / (1 + alpha)
           
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    return _sample_her_transitions, _sample_nstep_her_transitions, _sample_nstep_correct_her_transitions, \
             _sample_nstep_lambda_her_transitions , _sample_nstep_dynamic_her_transitions


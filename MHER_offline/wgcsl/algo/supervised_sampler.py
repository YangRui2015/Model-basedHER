from operator import methodcaller
from numpy.lib.function_base import bartlett
from wgcsl.common import logger
import numpy as np
from wgcsl.algo.util import random_log
from wgcsl.algo.adv_que import advque
#
from wgcsl.algo.dynamics import dynamic_interaction
from wgcsl.algo.util import get_ags_from_states

global global_threshold 
global_threshold = 0

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


def make_sample_transitions(replay_strategy, replay_k, reward_fun, no_relabel=False):
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  
        future_p = 0

    if no_relabel:
        print( '*' * 10 + 'Will not use relabeling in this method' + '*' * 10)

    
    def _get_future_R(episode_batch, episode_idxs, t_samples, batch_size, gamma):
        future_R = []
        for i in range(batch_size):
            epi_ag2 = episode_batch['ag_2'][episode_idxs[i]][t_samples[i]:]
            epi_g = episode_batch['g'][episode_idxs[i]][t_samples[i]:]
            rews = _get_reward(epi_ag2, epi_g)
            L = len(rews)
            R = pow(gamma, np.arange(L)).reshape(-1) * rews.reshape(-1)
            future_R.append(R.sum())
        return np.array(future_R)
    
    def _preprocess(episode_batch, batch_size_in_transitions, ags_std=None, use_ag_std=False):
        #  T = episode_batch['u'].shape[1] 
        T = episode_batch['u_2'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        if use_ag_std:
            episode_idxs = np.random.choice(np.arange(rollout_batch_size), batch_size, p=ags_std[:rollout_batch_size]/ags_std[:rollout_batch_size].sum())
        else:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        return transitions, episode_idxs, t_samples, batch_size, T

    def _get_reward(ag_2, g):
        info = {}
        reward_params = {'ag_2':ag_2, 'g':g}
        reward_params['info'] = info
        return reward_fun(**reward_params) + 1  # make rewards positive

    def _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=future_p, return_t=False):
        her_indexes = (np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T-t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        if not return_t:
            return future_ag.copy(), her_indexes.copy()
        else:
            return future_ag.copy(), her_indexes.copy(), future_offset
    
        
    def _reshape_transitions(transitions, batch_size, batch_size_in_transitions):
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions
    
    def _sample_her_transitions(episode_batch, batch_size_in_transitions, info=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        ags_std = None
        if info is not None and 'ags_std' in info.keys():
            ags_std = info['ags_std']

        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions, ags_std)
        if not no_relabel:
            future_ag, her_indexes = _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            if len(transitions['g'].shape) == 1:
                transitions['g'][her_indexes] = future_ag.reshape(-1)
            else:
                transitions['g'][her_indexes] = future_ag

        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)
    
    def _sample_conservative_transitions(episode_batch, batch_size_in_transitions, info):
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        random_action_fun, get_Q = info['random_action_fun'], info['get_Q']
        if not no_relabel:
            future_ag, her_indexes = _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            if len(transitions['g'].shape) == 1:
                transitions['g'][her_indexes] = future_ag.reshape(-1)
            else:
                transitions['g'][her_indexes] = future_ag

        actions, Qs = [], []
        negative_actions = []
        N = 20
        actions = random_action_fun(N)
        for i in range(N):
            Qs.append(get_Q(o=transitions['o'], u=np.array(actions[i]).repeat(batch_size, axis=0), g=transitions['g']))
        all_Qs = np.array(Qs).reshape((batch_size, N))
        actions = np.array(actions).reshape(-1, actions[0].shape[-1])
        for i in range(batch_size):
            neg_act = np.random.choice(np.arange(N), p=np.exp(all_Qs[i]) / np.exp(all_Qs[i]).sum())
            negative_actions.append(actions[neg_act])
        transitions['neg_u'] = np.array(negative_actions).reshape((batch_size, transitions['u'].shape[-1]))
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)
    

    def _sample_model_based_her_transitions(episode_batch, batch_size_in_transitions, info):
        steps, alpha = 10, info['alpha']
        dynamic_model, action_fun, obs_to_goal = info['dynamic_model'], info['action_fun'], info['obs_to_goal']
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        # update dynamic model
        loss = dynamic_model.update(transitions['o'], transitions['u'], transitions['o_2'], times=2)  

        random_log('using mher with alpha {}, step {}'.format(alpha, steps))
        if np.random.random() < 0.05:
            print('error: max {}, mean {}'.format(loss.max(), loss.mean()))

        relabel_indexes = (np.random.uniform(size=batch_size) < 0.8)
        # model-based relabeling
        last_state = transitions['o_2'].copy()  
        next_states_list = dynamic_interaction(last_state, transitions['g'], action_fun, dynamic_model, steps, act_noise=0.2)
        next_states_list.insert(0, last_state.copy())
        next_states_array = np.concatenate(next_states_list,axis=1).reshape(batch_size, steps+1, -1) 
        # her goals
        future_ag, _ = _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=1)
        step_idx = np.random.randint(next_states_array.shape[1], size=(batch_size))
        last_state = next_states_array[np.arange(batch_size).reshape(-1), step_idx]
        # add dynamic achieve goals
        new_ags, _ = get_ags_from_states(obs_to_goal, batch_size, last_state, 1)
        transitions['g'][relabel_indexes] = new_ags[relabel_indexes].reshape((relabel_indexes.sum(), -1)) 
        # mix future goals
        her_indexes = (np.random.uniform(size=batch_size) < 0.4) 
        transitions['g'][relabel_indexes & her_indexes] = future_ag[relabel_indexes & her_indexes]
        
        # recompute rewards
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g']) 
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)
    

    def _sample_supervised_transitions(episode_batch, batch_size_in_transitions, info):
        train_policy, gamma, get_Q_pi, method, get_ags_std  = info['train_policy'], info['gamma'], info['get_Q_pi'], info['method'], info['get_ags_std']
        ags_std = get_ags_std()
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions, ags_std, use_ag_std=False)
        random_log('using supervide policy learning with method {} and no relabel {}'.format(method, no_relabel))
        original_g = transitions['g'].copy() # save to train the value function
        if not no_relabel:
            future_ag, her_indexes, offset = _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=1, return_t=True) 
            transitions['g'][her_indexes] = future_ag
        else:
            offset = np.zeros(batch_size)

        if method == '':
            loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'])   # do not use weights
        else:
            method_lis = method.split('_')
            if 'gamma' in method_lis:
                weights = pow(gamma, offset)  
            else:
                weights = np.ones(batch_size)

            if 'AWR' in method_lis:
                future_R = _get_future_R(episode_batch, episode_idxs, t_samples, batch_size, gamma)
                value = get_Q_pi(o=transitions['o_2'], g=transitions['g']).reshape(-1)
                adv = future_R  - value
                weights *= np.exp(adv)

            if 'adv' in method_lis:
                value = get_Q_pi(o=transitions['o'], g=transitions['g']).reshape(-1)
                next_value = get_Q_pi(o=transitions['o_2'], g=transitions['g']).reshape(-1)
                adv = _get_reward(transitions['ag_2'], transitions['g']) + gamma * next_value  - value

                if not no_relabel: # no relabel refers to marvil, it has no this weight
                    advque.update(adv)
                    global global_threshold
                    global_threshold = min(global_threshold + 0.1, 80)  # 0.01 for difficult tasks and 0.1 for simple tasks
                    threshold = advque.get(global_threshold)
                    if np.random.random() < 0.1:
                        print(global_threshold, threshold, (adv >= threshold).mean())

                if 'exp' in method_lis:
                    if 'clip10' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 10)
                    elif 'clip5' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 5)
                    elif 'clip1' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 1)
                    else:
                        weights *= np.exp(adv) # exp weights
                else:
                    weights *= adv

                if not no_relabel:
                    positive = adv.copy()
                    positive[adv >= threshold] = 1
                    positive[adv < threshold] = 0.05
                    weights *= positive

            loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'], weights=weights)  

        # To train value function
        keep_origin_rate = 0.2
        origin_index = (np.random.uniform(size=batch_size) < keep_origin_rate)
        transitions['g'][origin_index] = original_g[origin_index]
        if method == 'AWR':
            transitions['r'] = future_R
        else:
            transitions['r'] = _get_reward(transitions['ag_2'], transitions['g']) 
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    return _sample_supervised_transitions, _sample_her_transitions, _sample_conservative_transitions, _sample_model_based_her_transitions


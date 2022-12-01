from stable_baselines.trpo_mpi import TRPO
# from stable_baselines.sac import SAC
from stable_baselines.mdpo import MDPO_OFF, MDPO_ON

import os
import warnings
from typing import Dict
import copy
import pickle

import cv2  # pytype:disable=import-error
import numpy as np
from gym import spaces

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecFrameStack
from stable_baselines.common.base_class import _UnvecWrapper



class MDAL_TRPO(TRPO):
    """
    Generative Adversarial Imitation Learning (GAIL)

    .. warning::

        Images are not yet handled properly by the current implementation


    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param expert_dataset: (ExpertDataset) the dataset manager
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param hidden_size: ([int]) the hidden dimension for the MLP
    :param g_step: (int) number of steps to train policy in each epoch
    :param d_step: (int) number of steps to train discriminator in each epoch
    :param d_stepsize: (float) the reward giver stepsize
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    # def __init__(self, policy, env, expert_dataset=None,
    #              hidden_size_adversary=100, adversary_entcoeff=1e-3, timesteps_per_batch=2000,
    #              g_step=5, d_step=1, d_stepsize=3e-4, verbose=0,
    #              _init_setup_model=True, **kwargs):
    def __init__(self, policy, env, expert_dataset=None,
                 hidden_size_adversary=100, adversary_entcoeff=0, # timesteps_per_batch=2000,
                 g_step=3, d_step=1, d_stepsize=3e-4, verbose=0,
                 _init_setup_model=True, exploration_bonus=False, bonus_coef=0.01, is_action_features=True,
                 neural=False, sgd_steps=1, lipschitz=1.0, **kwargs):
        super().__init__(policy, env, verbose=verbose, _init_setup_model=False, **kwargs)
        self.using_mdal = True
        self.expert_dataset = expert_dataset
        self.g_step = g_step
        self.d_step = d_step
        self.d_stepsize = d_stepsize
        self.hidden_size_adversary = hidden_size_adversary
        # self.timesteps_per_batch = timesteps_per_batch
        self.adversary_entcoeff = adversary_entcoeff
        self.exploration_bonus = exploration_bonus
        self.bonus_coef = bonus_coef
        self.is_action_features = is_action_features
        self.neural = neural
        self.sgd_steps = sgd_steps
        self.lipschitz = lipschitz


        if _init_setup_model:
            self.setup_model()

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="MDAL",
              reset_num_timesteps=True):
        assert self.expert_dataset is not None, "You must pass an expert dataset to MDAL for training"
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps)


class MDAL_MDPO_ON(MDPO_ON):
    """
    Generative Adversarial Imitation Learning (GAIL)

    .. warning::

        Images are not yet handled properly by the current implementation


    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param expert_dataset: (ExpertDataset) the dataset manager
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param hidden_size: ([int]) the hidden dimension for the MLP
    :param g_step: (int) number of steps to train policy in each epoch
    :param d_step: (int) number of steps to train discriminator in each epoch
    :param d_stepsize: (float) the reward giver stepsize
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """


    def __init__(self, policy, env, expert_dataset=None,
                 hidden_size_adversary=100, adversary_entcoeff=0, timesteps_per_batch=2000,
                 g_step=1, d_step=1, d_stepsize=3e-4, verbose=0,
                 _init_setup_model=True, exploration_bonus=False, bonus_coef=0.01, is_action_features=True,
                 neural=False, **kwargs):


        super().__init__(policy, env, verbose=verbose, _init_setup_model=False, **kwargs)
        self.using_mdal = True
        self.using_gail = False

        self.expert_dataset = expert_dataset
        self.g_step = g_step
        self.d_step = d_step
        self.d_stepsize = d_stepsize
        self.hidden_size_adversary = hidden_size_adversary
        self.timesteps_per_batch = timesteps_per_batch
        self.adversary_entcoeff = adversary_entcoeff
        self.exploration_bonus = exploration_bonus
        self.bonus_coef = bonus_coef
        self.is_action_features = is_action_features
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.neural = neural

        if _init_setup_model:
            self.setup_model()

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="MDAL",
              reset_num_timesteps=True):
        assert self.expert_dataset is not None, "You must pass an expert dataset to MDAL for training"
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps)


class MDAL_MDPO_OFF(MDPO_OFF):
    """
    Generative Adversarial Imitation Learning (GAIL)

    .. warning::

        Images are not yet handled properly by the current implementation


    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param expert_dataset: (ExpertDataset) the dataset manager
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param hidden_size: ([int]) the hidden dimension for the MLP
    :param g_step: (int) number of steps to train policy in each epoch
    :param d_step: (int) number of steps to train discriminator in each epoch
    :param d_stepsize: (float) the reward giver stepsize
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    # def __init__(self, policy, env, expert_dataset=None,
    #              hidden_size_adversary=100, adversary_entcoeff=1e-3, timesteps_per_batch=2000,
    #              g_step=5, d_step=1, d_stepsize=3e-4, verbose=0,
    #              _init_setup_model=True, **kwargs):

    def __init__(self, policy, env, expert_dataset=None,
                 hidden_size_adversary=100, adversary_entcoeff=0, timesteps_per_batch=2000,
                 g_step=1, d_step=10, d_stepsize=3e-4, verbose=0,
                 _init_setup_model=True, exploration_bonus=False, bonus_coef=0.01, is_action_features=True,
                  neural=False, lipschitz=1.0, **kwargs):
        super().__init__(policy, env, verbose=verbose, _init_setup_model=False, **kwargs)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.using_gail = False
        self.using_mdal = True
        self.expert_dataset = expert_dataset
        self.g_step = g_step
        self.d_step = d_step
        self.d_stepsize = d_stepsize
        self.hidden_size_adversary = hidden_size_adversary
        self.timesteps_per_batch = timesteps_per_batch
        self.adversary_entcoeff = adversary_entcoeff
        self.exploration_bonus = exploration_bonus
        self.bonus_coef = bonus_coef
        self.is_action_features = is_action_features
        self.neural = neural
        self.lipschitz = lipschitz
        if _init_setup_model:
            self.setup_model()

    def learn(self, total_timesteps, callback=None, log_interval=2000, tb_log_name="MDAL",
              reset_num_timesteps=True):
        assert self.expert_dataset is not None, "You must pass an expert dataset to MDAL for training"
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps)


class Proj_MDPO_OFF(MDPO_OFF):
    """
    Generative Adversarial Imitation Learning (GAIL)

    .. warning::

        Images are not yet handled properly by the current implementation


    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param expert_dataset: (ExpertDataset) the dataset manager
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param hidden_size: ([int]) the hidden dimension for the MLP
    :param g_step: (int) number of steps to train policy in each epoch
    :param d_step: (int) number of steps to train discriminator in each epoch
    :param d_stepsize: (float) the reward giver stepsize
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    # def __init__(self, policy, env, expert_dataset=None,
    #              hidden_size_adversary=100, adversary_entcoeff=1e-3, timesteps_per_batch=2000,
    #              g_step=5, d_step=1, d_stepsize=3e-4, verbose=0,
    #              _init_setup_model=True, **kwargs):

    def __init__(self, policy, env, expert_dataset=None,
                 hidden_size_adversary=100, adversary_entcoeff=0, timesteps_per_batch=2000,
                 g_step=1, d_step=10, d_stepsize=3e-4, verbose=0,
                 _init_setup_model=True, exploration_bonus=False, bonus_coef=0.01, is_action_features=True,
                  neural=False, lipschitz=1.0, **kwargs):
        super().__init__(policy, env, verbose=verbose, _init_setup_model=False, **kwargs)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.using_gail = False
        self.using_mdal = False
        self.using_proj = True
        self.expert_dataset = expert_dataset
        self.g_step = g_step
        self.d_step = d_step
        self.d_stepsize = d_stepsize
        self.hidden_size_adversary = hidden_size_adversary
        self.timesteps_per_batch = timesteps_per_batch
        self.adversary_entcoeff = adversary_entcoeff
        self.exploration_bonus = exploration_bonus
        self.bonus_coef = bonus_coef
        self.is_action_features = is_action_features
        self.neural = neural
        self.lipschitz = lipschitz
        if _init_setup_model:
            self.setup_model()

    def learn(self, total_timesteps, callback=None, log_interval=2000, tb_log_name="Proj_MDPO_OFF",
              reset_num_timesteps=True):
        assert self.expert_dataset is not None, "You must pass an expert dataset to Proj_MDPO_OFF for training"
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps)


class Proj(object):
    def __init__(self, policy, env, expert_dataset=None,
                 hidden_size_adversary=100, adversary_entcoeff=0, timesteps_per_batch=2000,
                 g_step=1, d_step=10, d_stepsize=3e-4, verbose=0,
                 _init_setup_model=True, exploration_bonus=False, bonus_coef=0.01, is_action_features=True,
                  neural=False, lipschitz=1.0, **kwargs):
        self.kwargs = {
            'policy': policy,
            'env': env,
            'expert_dataset': expert_dataset,
            'hidden_size_adversary': hidden_size_adversary,
            'adversary_entcoeff': adversary_entcoeff,
            'timesteps_per_batch': timesteps_per_batch,
            'g_step': g_step,
            'd_step': d_step,
            'd_stepsize': d_stepsize,
            'verbose': verbose,
            '_init_setup_model': _init_setup_model,
            'exploration_bonus': exploration_bonus,
            'bonus_coef': bonus_coef,
            'is_action_features': is_action_features,
            'neural': neural,
            'lipschitz': lipschitz,
        }
        self.kwargs.update(kwargs)

        self.expert_dataset = expert_dataset
        # self.normalize_s = np.array([1./2., 1./2., 1./16., 1./4.], dtype=np.float32)
        assert not is_action_features
        if env.envs[0].spec.id == 'Pendulum-v0':
            self.normalize_s = np.array([1./2., 1./2., 1./16.], dtype=np.float32)
            self.normalize_b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif env.envs[0].spec.id == 'MountainCarContinuous-v0':
            self.normalize_s = np.array([1/1.8, 1/0.14], dtype=np.float32)
            self.normalize_b = np.array([2./3., 0.5], dtype=np.float32)
        expert_feat_exp = self.expert_dataset.successor_features * 100 # gamma has to be 0.99. successor_features has extra * (1 - gamma)
        self.expert_feat_exp = expert_feat_exp[:-1] * self.normalize_s + self.normalize_b
        self.model = None
        self.alpha = None
        self.logdir = kwargs['tensorboard_log']
        if not os.path.exists(kwargs['tensorboard_log']):
            os.makedirs(kwargs['tensorboard_log'])

    def get_feature_expectation_raw(self, model, env, n_episodes=10):
        # Retrieve the environment using the RL model
        if env is None and isinstance(model, BaseRLModel):
            env = model.get_env()

        assert env is not None, "You must set the env in the model or pass it to the function."

        is_vec_env = False
        if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
            is_vec_env = True
            if env.num_envs > 1:
                warnings.warn("You are using multiple envs, only the data from the first one will be recorded.")

        # Sanity check
        assert (isinstance(env.observation_space, spaces.Box) or
                isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

        assert (isinstance(env.action_space, spaces.Box) or
                isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

        actions = []
        observations = []
        rewards = []
        episode_returns = np.zeros((n_episodes,))

        episode_starts = []
        gamma = model.gamma

        ep_idx = 0
        h_step = 0
        obs = env.reset()
        episode_obs = []
        episode_obs.append([])
        episode_gammas = []
        episode_gammas.append([])
        episode_act = []
        episode_act.append([])
        episode_starts.append(True)
        reward_sum = 0.0
        idx = 0
        # state and mask for recurrent policies
        state, mask = None, None

        if is_vec_env:
            mask = [True for _ in range(env.num_envs)]

        while ep_idx < n_episodes:
            obs_ = obs[0] if is_vec_env else obs
            observations.append(obs_)
            episode_obs[ep_idx].append(obs_)

            if isinstance(model, BaseRLModel):
                action, state = model.predict(obs, state=state, mask=mask)
            else:
                action = model(obs)

            obs, reward, done, _ = env.step(action)

            # Use only first env
            if is_vec_env:
                mask = [done[0] for _ in range(env.num_envs)]
                action = np.array([action[0]])
                reward = np.array([reward[0]])
                done = np.array([done[0]])

            actions.append(action)
            episode_gammas[ep_idx].append(gamma ** h_step)
            episode_act[ep_idx].append(action)
            rewards.append(reward)
            episode_starts.append(done)
            reward_sum += reward
            idx += 1
            h_step += 1
            if done:
                if not is_vec_env:
                    obs = env.reset()
                    # Reset the state in case of a recurrent policy
                    state = None
                episode_returns[ep_idx] = reward_sum
                reward_sum = 0.0
                h_step = 0
                ep_idx += 1
                if ep_idx < n_episodes:
                    episode_obs.append([])
                    episode_act.append([])
                    episode_gammas.append([])



        if isinstance(env.observation_space, spaces.Box):
            observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
        elif isinstance(env.observation_space, spaces.Discrete):
            observations = np.array(observations).reshape((-1, 1))

        if isinstance(env.action_space, spaces.Box):
            actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
        elif isinstance(env.action_space, spaces.Discrete):
            actions = np.array(actions).reshape((-1, 1))



        for ep_idx, (ep_obs, ep_act), in enumerate(zip(episode_obs, episode_act)):
            for idx, (obs, act) in enumerate(zip(reversed(ep_obs), reversed(ep_act))):
                current_features = np.concatenate((obs, act), axis=0)
                if idx == 0:
                    successor_features = (1-gamma) * current_features
                else:
                    successor_features = np.add(gamma * successor_features, (1 - gamma) * current_features)
            if ep_idx == 0:
                sum_successor_features = successor_features
            else:
                sum_successor_features = np.add(sum_successor_features, successor_features)

        successor_features = sum_successor_features / n_episodes

        rewards = np.array(rewards)
        episode_starts = np.array(episode_starts[:-1])

        assert len(observations) == len(actions)

        return successor_features

    def get_feature_expectation(self, model, env, n_episodes=10):
        feat_exp = self.get_feature_expectation_raw(model, env.envs[0].env, n_episodes)
        feat_exp = feat_exp * 100 # gamma has to be 0.99. successor_features has extra * (1 - gamma)
        feat_exp = feat_exp[:-1] * self.normalize_s + self.normalize_b
        return feat_exp

    def learn(self, total_timesteps, num_proj_iters=10, callback=None, log_interval=2000, tb_log_name="Proj_MDPO_OFF",
              reset_num_timesteps=True):
        assert self.expert_dataset is not None, "You must pass an expert dataset to Proj_MDPO_OFF for training"

        feat_exps_bar = []
        feat_exps = []
        ts = []
        policy_weights = []
        env = self.kwargs['env']

        # Initialize
        # kwargs = copy.deepcopy(self.kwargs)
        # kwargs['tensorboard_log'] = os.path.join(kwargs['tensorboard_log'], 'iter%04d' % 0)
        model = Proj_MDPO_OFF(**self.kwargs)
        feat_exp = self.get_feature_expectation(model, env, 100)
        model.save(os.path.join(self.logdir, 'iter%04d' % 0))
        del model
        feat_exps.append(feat_exp)
        feat_exps_bar.append(feat_exp)
        policy_weights.append(np.array([1.], dtype=np.float32))

        for i in range(1, num_proj_iters + 1):
            print('proj iter:', i)
            # kwargs = copy.deepcopy(self.kwargs)
            # kwargs['tensorboard_log'] = os.path.join(kwargs['tensorboard_log'], 'iter%04d' % i)
            model = Proj_MDPO_OFF(**self.kwargs)
            model.reward_giver.update_reward(self.expert_feat_exp - feat_exps_bar[-1])
            model.learn(total_timesteps // num_proj_iters, tb_log_name='iter%04d' % i)
            feat_exp = self.get_feature_expectation(model, env, 100)
            model.save(os.path.join(self.logdir, 'iter%04d' % i))
            del model
            feat_exps.append(feat_exp)

            alpha = np.dot(feat_exp - feat_exps_bar[-1], self.expert_feat_exp - feat_exps_bar[-1]) / \
                    np.dot(feat_exp - feat_exps_bar[-1], feat_exp - feat_exps_bar[-1])
            alpha = np.clip(alpha, 0, 1)
            feat_exp_bar = feat_exps_bar[-1] + alpha * (feat_exp - feat_exps_bar[-1])
            feat_exps_bar.append(feat_exp_bar)

            policy_weight_pre = np.concatenate([policy_weights[-1], np.array([0.], dtype=np.float32)], axis=0)
            policy_weight_i = np.zeros((i + 1,), dtype=np.float32)
            policy_weight_i[-1] = 1
            assert policy_weight_pre.shape == policy_weight_i.shape
            policy_weight = policy_weight_pre + alpha * (policy_weight_i - policy_weight_pre)
            policy_weights.append(policy_weight)

        with open(os.path.join(self.logdir, 'data.pkl'), 'wb') as f:
            pickle.dump({'feat_exps_bar': feat_exps_bar, 'policy_weights': policy_weights, 'feat_exps': feat_exps}, f)

    def save(self, save_path):
        pass

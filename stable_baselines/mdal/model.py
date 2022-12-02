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
                  neural=False, lipschitz=1.0, mdpSolver=None, **kwargs):
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
        self.mdpSolver = mdpSolver
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
        if self.mdpSolver == "PG":
            model = PG(**self.kwargs)
        else:
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
            if self.mdpSolver == "PG":
                model = PG(**self.kwargs)
            else:
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


class MWAL_MDPO_OFF(MDPO_OFF):
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
        self.using_proj = False
        self.using_mwal = True
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

    def learn(self, total_timesteps, callback=None, log_interval=2000, tb_log_name="MWAL_MDPO_OFF",
              reset_num_timesteps=True):
        assert self.expert_dataset is not None, "You must pass an expert dataset to MWAL_MDPO_OFF for training"
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps)


class MWAL(object):
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

    def learn(self, total_timesteps, T=10, callback=None, log_interval=2000, tb_log_name="MWAL_MDPO_OFF",
              reset_num_timesteps=True):
        assert self.expert_dataset is not None, "You must pass an expert dataset to MWAL_MDPO_OFF for training"

        feat_exps_bar = []
        feat_exps = []
        
        ts = []
        reward_vecs = []
        env = self.kwargs['env']

        # Initialize
        # kwargs = copy.deepcopy(self.kwargs)
        # kwargs['tensorboard_log'] = os.path.join(kwargs['tensorboard_log'], 'iter%04d' % 0)
        model = MWAL_MDPO_OFF(**self.kwargs)
        feat_exp = self.get_feature_expectation(model, env, 100)
        W = np.ones(np.shape(feat_exp))
        model.save(os.path.join(self.logdir, 'iter%04d' % 0))
        del model
        feat_exps.append(feat_exp)

        for i in range(1, T + 1):
            print('mwal iter:', i)
            # kwargs = copy.deepcopy(self.kwargs)
            # kwargs['tensorboard_log'] = os.path.join(kwargs['tensorboard_log'], 'iter%04d' % i)
            model = MWAL_MDPO_OFF(**self.kwargs)
            reward_vec = W / np.sum(W)
            reward_vecs.append(reward_vec)
            model.reward_giver.update_reward(reward_vec)
            model.learn(total_timesteps // T, tb_log_name='iter%04d' % i)
            feat_exp = self.get_feature_expectation(model, env, 100)
            feat_exps.append(feat_exp)
            model.save(os.path.join(self.logdir, 'iter%04d' % i))
            gamma = model.gamma
            del model
            beta = 1 / (1 + np.sqrt(2 * np.log(np.shape(W)) / T))
            G = ((1 - gamma) * (feat_exp - self.expert_feat_exp) + 2 * np.ones(np.shape(W))) / 4
            W = W * beta ** G

        policy_weights = np.ones(T) / T

        with open(os.path.join(self.logdir, 'data.pkl'), 'wb') as f:
            pickle.dump({'feat_exps_bar': feat_exps, 'policy_weights': policy_weights, 'feat_exps': feat_exps}, f)

    def save(self, save_path):
        pass


from stable_baselines.sac import SAC
import tensorflow as tf
from stable_baselines.sac.policies import SACPolicy
from stable_baselines.common import tf_util, SetVerbosity
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.buffers import ReplayBuffer
class PG(SAC):
    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                learning_starts=100, train_freq=1, update_reward_freq=2000, batch_size=64,
                tau=0.005, ent_coef='auto', target_update_interval=1,
                gradient_steps=1, target_entropy='auto', action_noise=None,
                random_exploration=0.0, verbose=0, tensorboard_log=None,
                _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                seed=None, n_cpu_tf_sess=None, **kwargs):

        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)
        
        self.using_gail = False
        self.using_mdal = False
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.update_reward_freq = update_reward_freq
        self.batch_size = batch_size
        self.tau = tau
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        # GAIL Params
        self.hidden_size_adversary = 100
        self.adversary_entcoeff = 1e-3
        self.expert_dataset = None
        self.g_step = 1
        self.d_step = 1
        self.d_stepsize = 3e-4

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None

        if _init_setup_model:
            self.setup_model()

    def get_env(self):
        return self.env

    def setup_model(self):
    
        from stable_baselines.mdal.adversary import LinearReward

        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.reward_giver = LinearReward(self.env, self.sess, self.observation_space, self.action_space,
                                                    is_action_features=False)


                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probability of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False,
                                                                    reuse=True)

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Target for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the Gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss

                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=tf_util.get_trainable_vars('model/pi'))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = tf_util.get_trainable_vars('model/values_fn')

                    source_params = tf_util.get_trainable_vars("model/values_fn")
                    target_params = tf_util.get_trainable_vars("target/values_fn")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, policy_train_op, train_values_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('value_loss', value_loss)
                    tf.summary.scalar('entropy', self.entropy)
                    if ent_coef_loss is not None:
                        tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        tf.summary.scalar('ent_coef', self.ent_coef)

                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.target_params = tf_util.get_trainable_vars("target/values_fn")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

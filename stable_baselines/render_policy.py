import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.mujoco_wrappers import wrap_mujoco

import gym

from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines import bench, logger


from stable_baselines.mdal import MDAL_MDPO_OFF, MDAL_MDPO_ON, MDAL_TRPO, Proj, MWAL, ProjFWMethod
from stable_baselines.gail import ExpertDataset, generate_expert_traj
import os

import warnings
from typing import Dict

import cv2  # pytype:disable=import-error
import numpy as np
from gym import spaces

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecFrameStack
from stable_baselines.common.base_class import _UnvecWrapper


def render(model, save_path=None, env=None, n_timesteps=0, n_episodes=100, evaluate=False, image_folder='recorded_images'):
    """
    Train expert controller (if needed) and record expert trajectories.

    .. note::

        only Box and Discrete spaces are supported for now.

    :param model: (RL model or callable) The expert model, if it needs to be trained,
        then you need to pass ``n_timesteps > 0``.
    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param env: (gym.Env) The environment, if not defined then it tries to use the model
        environment.
    :param n_timesteps: (int) Number of training timesteps
    :param n_episodes: (int) Number of trajectories (episodes) to record
    :param image_folder: (str) When using images, folder that will be used to record images.
    :return: (dict) the generated expert trajectories.
    """
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

    # Check if we need to record images
    obs_space = env.observation_space
    # record_images = len(obs_space.shape) == 3 and obs_space.shape[-1] in [1, 3, 4] \
    #                 and obs_space.dtype == np.uint8
    # if record_images and save_path is None:
    #     warnings.warn("Observations are images but no save path was specified, so will save in numpy archive; "
    #                   "this can lead to higher memory usage.")
    #     record_images = False

    # if not record_images and len(obs_space.shape) == 3 and obs_space.dtype == np.uint8:
    #     warnings.warn("The observations looks like images (shape = {}) "
    #                   "but the number of channel > 4, so it will be saved in the numpy archive "
    #                   "which can lead to high memory usage".format(obs_space.shape))
    record_images = True

    image_ext = 'jpg'
    if record_images:
        # We save images as jpg or png, that have only 3/4 color channels
        if isinstance(env, VecFrameStack) and env.n_stack == 4:
            # assert env.n_stack < 5, "The current data recorder does no support"\
            #                          "VecFrameStack with n_stack > 4"
            image_ext = 'png'

        folder_path = save_path #os.path.dirname(save_path)
        image_folder = os.path.join(folder_path, image_folder)
        os.makedirs(image_folder, exist_ok=True)
        print("=" * 10)
        print("Images will be recorded to {}/".format(image_folder))
        print("Image shape: {}".format(obs_space.shape))
        print("=" * 10)

    if n_timesteps > 0 and isinstance(model, BaseRLModel):
        model.learn(n_timesteps)

    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))

    episode_starts = []
    gamma = model.gamma

    ep_idx = 0
    h_step = 0
    obs = env.reset()
    img = env.render('rgb_array')
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
        if record_images:
            image_path = os.path.join(image_folder, "{}.{}".format(idx, image_ext))
            # Convert from RGB to BGR
            # which is the format OpenCV expect
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, img)
            observations.append(image_path)
        else:
            observations.append(obs_)
            episode_obs[ep_idx].append(obs_)

        if isinstance(model, BaseRLModel):
            action, state = model.predict(obs, state=state, mask=mask)
        else:
            action = model(obs)

        obs, reward, done, _ = env.step(action)
        img = env.render('rgb_array')

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

    env.close()


def test(env_id, algo, num_timesteps, seed, sgd_steps, t_pi, t_c, lam, log, expert_path, pretrain, pretrain_epochs,
         mdpo_update_steps, num_trajectories, expert_model, exploration_bonus, bonus_coef, random_action_len,
         is_action_features, dir_name, neural, lipschitz, mdpSolver, args):

    with tf_util.single_threaded_session():
        env_name = env_id[:-3].lower()

        def make_env():
            # env_out = gym.make(env_id, reset_noise_scale=1.0)
            env_out = gym.make(env_id)#, render_mode="rgb_array")
            # env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
            env_out.seed(seed)
            env_out = wrap_mujoco(env_out, random_action_len=random_action_len)
            return env_out

        env = DummyVecEnv([make_env])

        expert_path = './experts/' + expert_path
        expert_path = expert_path + '.npz'
        dataset = ExpertDataset(expert_path=expert_path, traj_limitation=2000, verbose=1)

        if algo == 'MDAL':
            model = MDAL_MDPO_OFF('MlpPolicy', env, dataset, verbose=1,
                                    tensorboard_log="./experiments_test/" + env_name + "/mdal/", seed=seed,
                                    buffer_size=1000000, ent_coef=0.0, learning_starts=10000, batch_size=256, tau=0.01,
                                    gamma=0.99, gradient_steps=sgd_steps, mdpo_update_steps=mdpo_update_steps,
                                    lam=0.0, train_freq=1, d_step=10, tsallis_q=1, reparameterize=True, t_pi=t_pi, t_c=t_c,
                                    exploration_bonus=exploration_bonus, bonus_coef=bonus_coef,
                                    is_action_features=is_action_features,
                                    neural=neural, lipschitz=lipschitz)
        elif algo == 'PROJ':
                model = Proj('MlpPolicy', env, dataset, verbose=1,
                                      tensorboard_log="./experiments_test/" + env_name + "/proj/", seed=seed,
                                      buffer_size=1000000, ent_coef=0.0, learning_starts=10000, batch_size=256, tau=0.01,
                                      gamma=0.99, gradient_steps=sgd_steps, mdpo_update_steps=mdpo_update_steps,
                                      lam=0.0, train_freq=1, d_step=10, tsallis_q=1, reparameterize=True, t_pi=t_pi, t_c=t_c,
                                      exploration_bonus=exploration_bonus, bonus_coef=bonus_coef,
                                      is_action_features=is_action_features,
                                      neural=neural, lipschitz=lipschitz, mdpSolver=mdpSolver)
                model = model.load_model(expert_model)
        elif algo == 'PROJ_FW':
            model = ProjFWMethod('MlpPolicy', env, dataset, verbose=1,
                                    tensorboard_log="./experiments_test/" + env_name + "/proj_fw/", seed=seed,
                                    buffer_size=1000000, ent_coef=0.0, learning_starts=10000, batch_size=256, tau=0.01,
                                    gamma=0.99, gradient_steps=sgd_steps, mdpo_update_steps=mdpo_update_steps,
                                    lam=0.0, train_freq=1, d_step=10, tsallis_q=1, reparameterize=True, t_pi=t_pi, t_c=t_c,
                                    exploration_bonus=exploration_bonus, bonus_coef=bonus_coef,
                                    is_action_features=is_action_features,
                                    neural=neural, lipschitz=lipschitz)
        elif algo == 'SAC':
            from stable_baselines import SAC
            env = VecNormalize(env, norm_reward=False, norm_obs=False)
            model = SAC.load(expert_model, env)

        render(model, '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/plots/%s_%s' % (algo, env_name), n_episodes=1)


if __name__ == '__main__':
    args = mujoco_arg_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    log = not args.no_log
    is_action_features = not args.states

    test(args.env, algo=args.algo, num_timesteps=args.num_timesteps, seed=args.seed_offset,
        expert_model=args.expert_model, expert_path=args.expert_path, num_trajectories=args.num_trajectories,
        is_action_features=is_action_features,
        sgd_steps=args.sgd_steps, mdpo_update_steps=args.mdpo_update_steps, lipschitz=args.lipschitz,
        t_pi=args.t_pi, t_c=args.t_c, lam=args.lam, log=log,
        pretrain=args.pretrain, pretrain_epochs=args.pretrain_epochs,
        exploration_bonus=args.exploration, bonus_coef=args.bonus_coef,
                    random_action_len=args.random_action_len, dir_name=args.dir_name, neural=args.neural, mdpSolver=args.mdpSolver,
                    args=args)

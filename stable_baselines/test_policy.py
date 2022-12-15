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


def test(env_id, algo, num_timesteps, seed, sgd_steps, t_pi, t_c, lam, log, expert_path, pretrain, pretrain_epochs,
         mdpo_update_steps, num_trajectories, expert_model, exploration_bonus, bonus_coef, random_action_len,
         is_action_features, dir_name, neural, lipschitz, mdpSolver, args):

    with tf_util.single_threaded_session():
        env_name = env_id[:-3].lower()

        def make_env():
            # env_out = gym.make(env_id, reset_noise_scale=1.0)
            env_out = gym.make(env_id)
            env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
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
            logdir = "./experiments/" + env_name + "/proj/"
            num_iters = 6
        elif algo == 'MWAL':
            model = MWAL('MlpPolicy', env, dataset, verbose=1,
                                    tensorboard_log="./experiments_test/" + env_name + "/mwal/", seed=seed,
                                    buffer_size=1000000, ent_coef=0.0, learning_starts=10000, batch_size=256, tau=0.01,
                                    gamma=0.99, gradient_steps=sgd_steps, mdpo_update_steps=mdpo_update_steps,
                                    lam=0.0, train_freq=1, d_step=10, tsallis_q=1, reparameterize=True, t_pi=t_pi, t_c=t_c,
                                    exploration_bonus=exploration_bonus, bonus_coef=bonus_coef,
                                    is_action_features=is_action_features,
                                    neural=neural, lipschitz=lipschitz)
            logdir = "./experiments/" + env_name + "/mwal/"
            num_iters = 7
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

        model.test(logdir, logdir, num_timesteps, num_iters)


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

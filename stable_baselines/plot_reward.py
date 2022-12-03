import os
import numpy as np
import matplotlib.pyplot as plt


env_name = 'MountainCarContinuous-v0'
#'Pendulum-v0'
if env_name == 'Pendulum-v0':
    data_paths = {
        'OAL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/mdal/tpi0.5_tc0.05_lam0.98_/MDAL_updateSteps10_states_only_s0/monitor.csv',
        'GAIL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/gail_mdpo_off/tpi0.5_tc0.05_lam0.98_/GAIL_MDPO_OFF_updateSteps10_states_only_s0/monitor.csv',
        'PROJ': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/proj/tpi0.5_tc0.05_lam0.98_/PROJ_updateSteps10_states_only_s0/monitor.csv',
    }
    gt_reward = 0
elif env_name == 'MountainCarContinuous-v0':
    data_paths = {
        'OAL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/mdal/tpi0.5_tc0.05_lam0.98_/MDAL_updateSteps10_states_only_s0/monitor.csv',
        'GAIL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/gail_mdpo_off/tpi0.5_tc0.05_lam0.98_/GAIL_MDPO_OFF_updateSteps10_states_only_s0/monitor.csv',
        'PROJ': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/proj/tpi0.5_tc0.05_lam0.98_/PROJ_updateSteps10_states_only_s0/monitor.csv',
    }
    gt_reward = 100
out_dir = '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/plots'
methods = ['PROJ', 'GAIL', 'OAL']
colors = ['red', 'blue', 'green']
running_mean_half = 20

# Plot
plt.figure(figsize=(8, 6))
for j, method in enumerate(methods):
    data = np.genfromtxt(data_paths[method], delimiter=',')
    rewards = data[1:, 0]
    steps = np.cumsum(data[1:, 1])
    rewards_runmean = []
    for i in range(rewards.shape[0]):
        start = max(0, i - running_mean_half)
        end = i + running_mean_half
        rewards_runmean.append(np.mean(rewards[start : end]))
    plt.plot(np.log(steps), np.array(rewards_runmean), label=method, c=colors[j])
    if method == 'PROJ':
        plt.plot(np.log(steps), np.zeros_like(np.array(rewards_runmean)) + gt_reward, c='black', linestyle='--')
plt.legend()
plt.xlabel('log number of policy updates')
plt.ylabel('true rewards')
plt.title(env_name + ': True Reward vs. Log Number of Policy Updates')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, env_name + '.jpg'))

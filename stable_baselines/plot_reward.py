import os
import numpy as np
import matplotlib.pyplot as plt


env_name = 'Pendulum-v0'
#'MountainCarContinuous-v0'
if env_name == 'Pendulum-v0':
    data_paths = {
        'OAL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/mdal/tpi0.5_tc0.05_lam0.98_/MDAL_updateSteps10_states_only_s0/monitor.csv',
        'GAIL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/gail_mdpo_off/tpi0.5_tc0.05_lam0.98_/GAIL_MDPO_OFF_updateSteps10_states_only_s0/monitor.csv',
        'PROJ': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/proj/tpi0.5_tc0.05_lam0.98_/PROJ_updateSteps10_states_only_s0/monitor.csv',
        'FWAS': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/proj_fw/tpi0.5_tc0.05_lam0.98_/PROJ_FW_updateSteps10_states_only_s0/monitor.csv',
    }
    gt_reward = -153.85
elif env_name == 'MountainCarContinuous-v0':
    data_paths = {
        'OAL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/mdal/tpi0.5_tc0.05_lam0.98_/MDAL_updateSteps10_states_only_s0/monitor.csv',
        'GAIL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/gail_mdpo_off/tpi0.5_tc0.05_lam0.98_/GAIL_MDPO_OFF_updateSteps10_states_only_s0/monitor.csv',
        'PROJ': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/proj/tpi0.5_tc0.05_lam0.98_/PROJ_updateSteps10_states_only_s0/monitor.csv',
        'FWAS': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/proj_fw/tpi0.5_tc0.05_lam0.98_/PROJ_FW_updateSteps10_states_only_s0/monitor.csv',
    }
    gt_reward = 94.47
out_dir = '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/plots'
methods = ['PROJ', 'FWAS', 'GAIL', 'OAL']
colors = ['red', 'orange', 'blue', 'green']
running_mean_half = 50

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
    if method == 'PROJ':
        plt.plot(steps, np.zeros_like(np.array(rewards_runmean)) + gt_reward, c='black', linestyle='--', label='Expert')
    plt.plot(steps, np.array(rewards_runmean), label=method, alpha=0.75)#, c=colors[j])
plt.legend()
plt.xscale('log')
plt.xlabel('log number of policy updates')
plt.ylabel('mean true rewards over %d episodes' % (2 * running_mean_half))
plt.title(env_name + ': True Reward vs. Log Number of Policy Updates')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, env_name + '.jpg'))

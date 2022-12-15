import os
import numpy as np
import matplotlib.pyplot as plt


env_name = 'Pendulum-v0'#'MountainCarContinuous-v0'#
if env_name == 'Pendulum-v0':
    data_paths = {
        'OAL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/mdal/tpi0.5_tc0.05_lam0.98_/MDAL_updateSteps10_s0/monitor.csv',
        'GAIL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/gail_mdpo_off/tpi0.5_tc0.05_lam0.98_/GAIL_MDPO_OFF_updateSteps10_states_only_s0/monitor.csv',
        'PROJ': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/proj/test.csv',#'/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/proj/tpi0.5_tc0.05_lam0.98_/PROJ_updateSteps10_states_only_s0/monitor.csv',
        'FWAS': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/proj_fw_backtracelinesearch/tpi0.5_tc0.05_lam0.98_/PROJ_FW_updateSteps10_states_only_s0/monitor.csv',
        'MWAL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/mwal/test.csv',#'/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/pendulum/mwal/tpi0.5_tc0.05_lam0.98_/MWAL_updateSteps10_states_only_s0/monitor.csv',
    }
    gt_reward = -153.85
elif env_name == 'MountainCarContinuous-v0':
    data_paths = {
        'OAL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/mdal/tpi0.5_tc0.05_lam0.98_/MDAL_updateSteps10_s0/monitor.csv',
        'GAIL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/gail_mdpo_off/tpi0.5_tc0.05_lam0.98_/GAIL_MDPO_OFF_updateSteps10_states_only_s0/monitor.csv',
        'PROJ': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/proj/test.csv',#'/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/proj/tpi0.5_tc0.05_lam0.98_/PROJ_updateSteps10_states_only_s0/monitor.csv',
        'FWAS': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/proj_fw_backtracelinesearch/tpi0.5_tc0.05_lam0.98_/PROJ_FW_updateSteps10_states_only_s0/monitor.csv',
        'MWAL': '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/mwal/test.csv',#'/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/experiments/mountaincarcontinuous/mwal/tpi0.5_tc0.05_lam0.98_/MWAL_updateSteps10_states_only_s0/monitor.csv',
    }
    gt_reward = 94.47
out_dir = '/mnt/Exp_ziqian/cmpt981/proj/OAL-cmpt981/stable_baselines/plots'
methods = ['PROJ', 'MWAL', 'OAL']#, 'GAIL', 'FWAS'
colors = ['red', 'pink', 'green']#, 'blue', 'orange'
running_mean_half = 50

# Plot
plt.figure(figsize=(8, 6))
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize

data = np.genfromtxt(data_paths['PROJ'], delimiter=',')
rewards = data[1:, 0]
steps = np.cumsum(data[1:, 1])
learn_start = 0
# while steps[learn_start] < (4000 if env_name == 'Pendulum-v0' else 10000):
#     learn_start += 1
# print(learn_start)
rewards = rewards[learn_start:]
steps = steps[learn_start:] / 10
plt.plot(steps, np.zeros_like(steps) + gt_reward, c='black', linestyle='--', label='Expert', linewidth=3)

for j, method in enumerate(methods):
    data = np.genfromtxt(data_paths[method], delimiter=',')
    rewards = data[1:, 0]
    steps = np.cumsum(data[1:, 1])
    learn_start = 0
    # while steps[learn_start] < (4000 if env_name == 'Pendulum-v0' else 10000):
    #     learn_start += 1
    # print(learn_start)
    rewards = rewards[learn_start:]
    steps = steps[learn_start:] / 10
    if method == 'OAL' or method == 'GAIL':
        rewards_runmean = []
        for i in range(rewards.shape[0]):
            start = max(0, i - running_mean_half)
            end = i + running_mean_half
            rewards_runmean.append(np.mean(rewards[start : end]))
    else:
        rewards_runmean = rewards
    plt.plot(steps, np.array(rewards_runmean), label=method, linewidth=3)#, alpha=0.75)#, c=colors[j])
plt.legend()#loc='upper left')
# plt.xscale('log')
plt.xlabel('number of policy updates')
plt.ylabel('mean true rewards over %d episodes' % (2 * running_mean_half))
plt.title(env_name + ': True Reward vs. Number of Policy Updates', fontsize=13.5)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, env_name + '.jpg'))

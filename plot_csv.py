import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats

color_list = ['r', 'b', 'g', 'orange', 'indigo','purple', 'black']

def plot_length_result():
    filetype = '.csv'
    file_list = []
    method_list = ['Ours', 'TARE Local', 'NBVP', 'Utility 10']
    method_param_list = ['ours_', 'TARE', 'NBVP', 'utility']
    legend_list = []

    for _, _, files in os.walk(f'results/length'):
        for f in files:
            if filetype in f:
                file_list.append(f)
    file_list.sort(reverse=True)

    results_list = []

    for i, param in enumerate(method_param_list):
        for csv_file in file_list:
            if param in csv_file:
                print(csv_file)
                csv_file = f'results/length/'+csv_file
                length_history = pd.read_csv(csv_file)
                results_list.append(length_history.to_numpy())
    results = np.array(results_list)
    # print(results.shape)
    wining_algorithm = np.argmin(results, axis=0)
    print(np.argwhere(wining_algorithm == 0))
    print(np.sum(wining_algorithm == 1))
    print(np.sum(wining_algorithm == 2))
    print(np.sum(wining_algorithm == 3))
    # ours = results[0].reshape(-1,1)
    # tare = results[1].reshape(-1,1)
    # nbvp = results[2].reshape(-1,1)
    # utility = results[3].reshape(-1,1)
    # t, pval = scipy.stats.ttest_ind(ours, tare)
    # print(t, pval)

def plot_trajectory_history_result():
    filetype = '.csv'
    file_list = []
    method_list = ['Ours', '$\lambda$ = 10', 'nearest']
    method_param_list = ['ours_', 'lambda10.', 'nearest']
    legend_list = []
    
    for _, _, files in os.walk(f'results/trajectory'):
        for f in files:
            if filetype in f:
                file_list.append(f)
    file_list.sort(reverse=True)

    fig = plt.figure(figsize=(3.8, 3))

    for i, param in enumerate(method_param_list):
        for csv_file in file_list:
            if param in csv_file:
                print(csv_file)
                csv_file = f'results/trajectory/'+csv_file
                trajectory_history = pd.read_csv(csv_file)
                trajectory_history = trajectory_history.sort_values('dist')
                num_data = len(trajectory_history)
                window_size = int(num_data / 75)
                print(window_size)

                trajectory_mean = trajectory_history.rolling(window_size, on='dist', center=False, min_periods=0).mean()
                trajectory_std = trajectory_history.rolling(window_size, on='dist', center=False, min_periods=0).std()
                line = plt.plot(trajectory_mean.dist, trajectory_mean.area, color=color_list[i], linewidth=2, zorder=10-i)
                plt.fill_between(trajectory_mean.dist, trajectory_mean.area-trajectory_std.area, trajectory_mean.area+trajectory_std.area, color=color_list[i], alpha=0.15)
                legend_list.append(line[0])

    plt.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2))
    plt.xlabel('Distance', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.ylabel("Explored Area", fontdict={'family': 'Times New Roman', 'size': 14})
    plt.xlim(0, 3000)
    plt.ylim(1e04, 9.5e04)
    plt.xticks(fontproperties='Times New Roman', size=12)
    plt.yticks(fontproperties='Times New Roman', size=12)
    plt.legend(legend_list, method_list, labelspacing=0.1, borderaxespad=0.1, handlelength=0.5, prop={'family':'Times New Roman', 'size':8}, frameon=False)
    # plt.show()
    plt.tight_layout()
    plt.savefig(f'results/trajectory_analysis.pdf')

if __name__ == "__main__":
    plot_length_result()
    # plot_trajectory_history_result()

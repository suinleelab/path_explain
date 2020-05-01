import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from line_plot import line_bar_plot

def main():
    datasets = os.listdir('../data/')
    datasets = [dataset.split('.')[0] for dataset in datasets if dataset.endswith('npz')]
    interaction_types = ['expected_hessians',
                         'hessians',
                         'shapley_sampling',
                         'contextual_decomposition',
                         'neural_interaction_detection']
    results_directories = ['results', 'results_random_draw']

    for result_dir in results_directories:
        for dataset in datasets:
            try:
                df_list = []
                base_dir = os.path.join('../', result_dir, dataset + '_{}.csv')
                for interaction_type in interaction_types:
                    df = pd.read_csv(base_dir.format(interaction_type))
                    df_list.append(df)
                df_total = pd.concat(df_list)
                plt.figure()
                plot = line_bar_plot(x='num_interactions_removed',
                                      y='mean_perf',
                                      data=df_total,
                                      color_by='interaction_type',
                                      ax=None,
                                      dpi=150,
                                      xlabel='Number of Interactions Removed',
                                      ylabel='Mean Sqaured Error',
                                      title='{}'.format(dataset.split('_')[-1] + ' interactions'),
                                      sd='sd_perf',
                                      legend_title='Interaction Method',
                                      loc='lower right')
                plt.savefig('{}_{}.png'.format(dataset, result_dir), dpi=150)
            except FileNotFoundError:
                print('Could not find: ' + base_dir.format(interaction_type))
                continue

if __name__ == '__main__':
    main()

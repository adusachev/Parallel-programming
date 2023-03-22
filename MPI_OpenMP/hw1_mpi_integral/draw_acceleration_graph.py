import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', font_scale=1.5)


data = pd.read_csv('results_4.csv')
data_sorted = data.sort_values(by=['N', ' p'])


labels = {1000: r'$10^3$', 1000000: r'$10^6$', 100000000: r'$10^8$'}
N_list = data_sorted['N'].values

fig = plt.figure(figsize=(12, 6))

for N in np.unique(N_list):
    indexes = np.where(N_list == N)[0]
    p = data_sorted[' p'].values[indexes]
    T1 = data_sorted[' T1'].values[indexes]
    Tp = data_sorted[' Tp'].values[indexes]
    S = T1 / Tp
    plt.plot(p, S, lw=3, label=f'N = {labels[N]}')
plt.xlabel('Количество процессов, p')
plt.ylabel('Ускорение, S(p)')
plt.legend()
plt.grid()

fig.savefig('./acceleration_graph_4.png')

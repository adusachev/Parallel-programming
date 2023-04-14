import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', font_scale=1.5)


data = pd.read_csv('results.csv')
data_sorted = data.sort_values(by=[' n_threads'])



p = data_sorted[' n_threads'].values
T1 = data_sorted['T_1'].values
Tp = data_sorted[' T_p'].values
S = T1 / Tp

fig = plt.figure(figsize=(12, 6))
plt.plot(p, S, lw=3)
plt.xlabel('Количество потоков, p')
plt.ylabel('Ускорение, S(p)')
plt.xticks(p)
plt.yticks(p)
plt.grid()

fig.savefig('./test_graph.png')

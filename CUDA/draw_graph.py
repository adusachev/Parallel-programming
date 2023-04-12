import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', font_scale=1.5)


data = pd.read_csv('results.csv')
data_sorted = data.sort_values(by=['Blocksize'])


blocksizes = data_sorted['Blocksize'].values
times = data_sorted[' Time'].values

fig = plt.figure(figsize=(12, 6))

plt.plot(blocksizes, times)

plt.xlabel('Block size')
plt.ylabel('Time, ms')
# plt.xticks(blocksizes)
plt.grid()

fig.savefig('./test_graph_matrixmul.png')

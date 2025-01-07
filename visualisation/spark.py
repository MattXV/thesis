import json
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import scienceplots


plt.rcParams["text.usetex"] = True
plt.style.use('science')

with open('legend.json', 'r') as file:
    legend = json.load(file)


df = pd.read_excel('absorption.xlsx')
means = dict()
for category in pd.unique(df.iloc[:, 0]):
    materials = np.array(df.loc[df['Category'] == category].iloc[:, 3:9])
    mean = np.mean(materials, axis=0)
    c_min = materials[np.argmin(np.sum(materials, axis=1)), :]
    c_max = materials[np.argmax(np.sum(materials, axis=1)), :]
    min_distance = np.subtract(mean, c_min)
    max_distance = np.subtract(mean, c_max)
    means[category] = {'mean': mean,
                       'min_distance': np.abs(min_distance), 
                       'max_distance': np.abs(max_distance),
                       'LOW': c_min,
                       'HIGH': c_max}

erb_ticks = [125, 250, 500, 1000, 2000, 4000]
# erb_ticks = ['{}'.format(i) for i in erb_ticks]
freqs = df.columns[3:].astype(np.int16).tolist()
for key, mat in sorted(means.items(), key=lambda i: np.sum(i[1]['mean'])):
    for mag in ['LOW', 'HIGH']:
        line = np.zeros((6, 2))
        for i, freq in enumerate(mat[mag]):
            line[i] = freqs[i], freq
        label = '{}_{}'.format(key, mag).lower()
        color =  np.array(legend[label.replace(' ', '_')]) / 255

        fig, ax = plt.subplots(1, 1, figsize=(10,3))
        ax.plot(line[:, 0], line[:, 1], label=label,
                    marker='o', color=color)
        for v in ax.spines.values():
            v.set_visible(False)
        ax.set_xscale('log')
        ax.set_ylim(-0.1, 1.1)
        ax.fill_between(freqs, line[:, 1], color=(*color, 0.3))
        # ax.set_xticks([])
        ax.set_xticks(erb_ticks)
        ax.set_xticklabels(erb_ticks, fontsize=32)
        ax.set_yticks([])
        fig.savefig(os.path.join(os.getcwd(), 'sparks', label + '.pdf'))
        quit()
        del fig, ax
    err = np.array([mat['min_distance'], mat['max_distance']])

    #colour = np.array(legend[label.replace(' ', '_')]) / 255
    # ax.errorbar(line[:, 0], line[:, 1], yerr=err, label=label,
    #             marker='o', capsize=7)

    


# x = np.cumsum(np.random.rand(1000)-0.5)

# # plot it
# fig, ax = plt.subplots(1,1,figsize=(10,3))
# plt.plot(x, color='k')
# plt.plot(len(x)-1, x[-1], color='r', marker='o')

# # remove all the axes

# #show it
# plt.show()
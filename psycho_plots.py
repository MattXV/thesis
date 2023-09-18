# %%
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import scienceplots
import numpy as np
import pandas as pd

plt.rcParams["text.usetex"] = True
plt.style.use('science')

# %%
t = np.arange(0, 4, 0.01)
a = np.sin(2 * np.pi * t)

pos = [0, 45, 90, 135, 180, 225, 270, 315]
dis = np.random.uniform(low=0.3, high=1, size=len(pos))
head = plt.imread('listener_head_topview.png')
head = OffsetImage(head, zoom=0.2)
ab = AnnotationBbox(head, (0, 0), xycoords='data', frameon=False)

fig, ax = plt.subplots(
    subplot_kw={'projection': 'polar'},
    figsize=(4, 4),
    dpi=300,
    tight_layout=True)
ax.plot(np.radians(pos), dis, 'ro')
ax.add_artist(ab)
ax.set_label('')

plt.show()

# %%
fig, ax = plt.subplots(
    subplot_kw={'projection': 'polar'},
    figsize=(4, 4),
    dpi=300,
    tight_layout=True)
# %%
localisation_df = pd.read_excel('filtered_loc.xlsx')

#%%
rt_df = localisation_df.loc[(localisation_df['RT'] == True)]
nort_df = localisation_df.loc[(localisation_df['RT'] == False)]

error_near    = nort_df.loc[(localisation_df['ArrayDistance'] == 'Near')]
error_mid     = nort_df.loc[(localisation_df['ArrayDistance'] == 'Medium')]
error_far     = nort_df.loc[(localisation_df['ArrayDistance'] == 'Far')]
error_near_rt = rt_df.loc[(localisation_df['ArrayDistance'] == 'Near')]
error_mid_rt  = rt_df.loc[(localisation_df['ArrayDistance'] == 'Medium')]
error_far_rt  = rt_df.loc[(localisation_df['ArrayDistance'] == 'Far')]

#%% PSPP export



# %%


fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 3.4))
ticks = [i * 5 for i in range(7)]
ticklabels = ['${}^\circ$'.format(i) for i in ticks]

ax.boxplot([error_near['AngularDistance'], error_mid['AngularDistance'], error_far['AngularDistance']],
           notch=False, showmeans=True, labels=['Near', 'Medium', 'Far'],
           flierprops=dict(markerfacecolor='g', marker='D'), widths=0.5, showfliers=False)
ax.set_title('HRTF')
ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels)

bx.boxplot([error_near_rt['AngularDistance'], error_mid_rt['AngularDistance'], error_far_rt['AngularDistance']],
           notch=False, showmeans=True, labels=['Near', 'Medium', 'Far'],
           flierprops=dict(markerfacecolor='g', marker='D'), widths=0.5, showfliers=False)
bx.set_title('HRTF + Pipeline')
bx.set_yticks(ticks)
bx.set_yticklabels(ticklabels)
ax.set_ylim(0, 32)
bx.set_ylim(0, 32)
fig.suptitle('Localisation Error', fontsize=14, y=0.93)
plt.show()

# %%
fig.savefig('localisation_accuracy.pdf', transparent=True)
# %%

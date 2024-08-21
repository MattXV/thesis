# %%
import scienceplots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["text.usetex"] = True
plt.style.use('science')

DISTANCE_CSV = 'results.csv'

# %%

distances = pd.read_csv(DISTANCE_CSV)

data = [distances.iloc[:, i].tolist() for i in range(len(distances.columns))]


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    


# %%

fig, ax = plt.subplots(tight_layout=True, figsize=(6.4, 5))

ax.set_title('Measured Perceptual Distances')
parts = ax.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=True)

for partname in ('cbars','cmins','cmaxes'):
    vp = parts[partname]
    vp.set_edgecolor('k')
    vp.set_linewidth(1)

for pc in parts['bodies']:
    pc.set_facecolor('tab:blue')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='tab:orange', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='tab:green', linestyle='-', lw=5)
# ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# set style for the axes
labels = ['Room', 'Office', 'Church', 'Village']
set_axis_style(ax, labels)
ax.set_xlabel('$D(Predicted, Truth)$')

plt.show()

fig.savefig('ism-lpips.pdf')
# %%


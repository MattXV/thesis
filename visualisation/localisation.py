# %%
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import scienceplots
import numpy as np

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
fig.savefig('localisation.png', transparent=True)
# %%

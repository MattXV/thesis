import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import scienceplots
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image


plt.rcParams["text.usetex"] = True
plt.style.use('science')


fig, (ax, bx) = plt.subplots(ncols=2, figsize=(6.4, 3))
fig.set_layout_engine('tight')

im = Image.open('character.png')

ax.imshow(im)
im = gaussian_filter(im, sigma=1)
bx.imshow(im)

# myInterval=4
# loc = plticker.MultipleLocator(base=myInterval)
# minloc = plticker.MultipleLocator(base=1)
# ax.xaxis.set_major_locator(loc)
# ax.yaxis.set_major_locator(loc)
# ax.xaxis.set_minor_locator(minloc)
# ax.yaxis.set_minor_locator(minloc)


# ax.set_xticks(np.arange(0, im.shape[0], 1))
# ax.set_yticks(np.arange(0, im.shape[1], 1))

# # Labels for major ticks
# ax.set_xticklabels(np.arange(0, im.shape[0], 1))
# ax.set_yticklabels(np.arange(0, im.shape[1], 1))

# # Minor ticks
ax.set_xticks(np.arange(-.5, im.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, im.shape[1], 1), minor=True)

ax.set_xticks(np.arange(-.5, im.shape[1], 4), [str(i) for i in np.arange(0, im.shape[0] + 1, 4)], minor=False)
ax.set_yticks(np.arange(-.5, im.shape[1], 4), [str(i) for i in np.arange(0, im.shape[1] + 1, 4)], minor=False)


ax.grid(which='major', axis='both', linestyle='-')
ax.grid(which='minor', axis='both', linestyle='--')

plt.show()
fig.savefig('image-processing-example.pdf')
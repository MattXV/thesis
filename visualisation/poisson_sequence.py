#%%
import math
import numpy as np
from random import uniform
import matplotlib.pyplot as plt
import scienceplots

plt.rcParams["text.usetex"] = True
plt.style.use('science')
# %%
def delta_t(z, t, v):
    u = (4 * math.pi * math.pow(343, 3) * math.pow(t, 2)) / v
    return math.log(1 / z) / u


def interval_t0(v):
    return math.pow((2 * v * math.log2(2)) / (4 * math.pi * math.pow(343, 3)), 1/3)


def poisson_dirac_sequence(energy_hist, hist_fs, volume):
    dirac_sequence = np.zeros_like(energy_hist)
    value = 1
    td = int(interval_t0(volume) * hist_fs)   
    while td < energy_hist.shape[0]:
        dirac_sequence[int(td)] = value
        interval = delta_t(uniform(0.01, 1), td / hist_fs, volume) * hist_fs
        td += max(interval, 1)

        value *= -1
    return dirac_sequence

# %%

fig, ax = plt.subplots(1, 1,  tight_layout=True, dpi=300, figsize=(6.4, 2.2))



x = np.linspace(0, 1, 44100)
x = x[0:2600]
y = poisson_dirac_sequence(x, 44100, 60)
ax.plot(x, y)

ax.set_yticks([-1, 0, 1])
ax.set_ylabel('Amplitude')
ax.set_xlabel('Time (s)')


plt.show()


# %%
fig.savefig('poisson-sequence.pdf', transparent=True)
# %%

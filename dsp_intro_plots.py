#%%
import matplotlib.pyplot as plt
# import mplcyberpunk
import numpy as np
from scipy import signal as sps
# plt.style.use('cyberpunk')

#%%

def rounder(values):
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]
    return np.frompyfunc(f, 1, 1)


#%% Basic sine
a = np.linspace(0, 1, 1024)
b = np.linspace(0, 1, 32)
f = 4
discrete_values = np.linspace(-1, 1, 9)
by = np.sin(b * f * np.pi)
y = np.sin(a * f * np.pi)
by = rounder(discrete_values)(by)

fig, (ax, bx) = plt.subplots(2)
fig.set_layout_engine('tight')

ax.plot(a, y)
ax.set_yticks([-1, 0, 1], [-1.0, 0, 1.0])
ax.tick_params(axis='both', length=0, width=0)
ax.set_title('Analogue Signal')

bx.stem(b, by, basefmt='k', markerfmt='ro')
bx.set_xticks([], minor=False)
bx.set_xticks(b, minor=True)
bx.set_yticks(discrete_values)
bx.grid(which='both', axis='y', linestyle='--')
bx.tick_params(axis='x', which='minor', length=6, width=2, colors='r')
# bx.grid(which='minor', axis='x', linestyle='-')
bx.set_title('Digital Signal')
fig.savefig('analogue_digital.png')
plt.show()

#%%
a = np.linspace(0, 1, 1024)
b = np.linspace(0, 1, 1024)
f1 = 4
f2 = 1
by = np.sin(b * f2 * np.pi)
y = np.sin(a * f1 * np.pi)

fig, ax = plt.subplots()
fig.set_layout_engine('tight')

ax.plot(a, y)
ax.plot(b, by)
ax.set_title('Analogue Signal')

plt.show()


# %% Nyquist-Shannon Theorem

f1 = 6
f2 = 3
a_x = np.linspace(0, 1, 1024)
b_x = np.linspace(0, 1, 1024)
a_y = np.sin(a_x * f1 * np.pi)
b_y = np.sin(b_x * f2 * np.pi)
# b_y = sps.resample(b_y, f1)
fig, ax = plt.subplots()

ax.plot(a_x, a_y)
ax.plot(b_x, b_y)
plt.show()
# %%

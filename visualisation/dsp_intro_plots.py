#%%
import matplotlib.pyplot as plt
# import mplcyberpunk
import numpy as np
from scipy import fft
# plt.style.use('cyberpunk')
plt.rcParams["text.usetex"] = True
plt.style.use('science')

#%%

def rounder(values):
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]
    return np.frompyfunc(f, 1, 1)


#%% Basic sine
a = np.linspace(0, 1, 1024)
b = np.linspace(0, 1, 16)
f = 2
discrete_values = np.linspace(-1, 1, 9)
by = np.sin(b * f * np.pi)
y = np.sin(a * f * np.pi)
by = rounder(discrete_values)(by)

fig, (ax, bx) = plt.subplots(1, 2, tight_layout=True, dpi=300, figsize=(6.4, 3))
fig.set_layout_engine('tight')

ax.plot(a, y)
ax.set_yticks([-1, 0, 1], [-1.0, 0, 1.0])
ax.tick_params(axis='both', length=0, width=0)
ax.set_title('Analogue Signal')

bx.stem(b, by, basefmt='k', markerfmt='ro')
bx.set_xticks([], minor=False)
bx.set_xticks(b, minor=True)
bx.set_yticks(discrete_values)
bx.grid(which='major', axis='y', linestyle='--')
bx.tick_params(axis='x', which='minor', length=6, width=2, colors='r')
# bx.grid(which='minor', axis='x', linestyle='-')
bx.set_title('Digital Signal')
fig.savefig('analogue_digital.pdf')
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


# %%
fs = 44100
dt = 1/44100
n = 64
a = np.linspace(0, 1, int(fs))
a_x = np.sin(a * 20 * np.pi)

fft_ax = fft.fft(a_x, n)
fft_t = np.arange(0, N, fs/)

fig, (ax, bx) = plt.subplots(2)
ax.plot(a, a_x)
bx.plot(fft_t, fft_ax.real)
plt.show()

# %%
fs = 44100
dt = 1/44100
n = 64
a = np.linspace(0, 1, int(fs))
a_x = np.sin(a * 20 * np.pi)

fft_ax = fft.fft(a_x, n)
fft_t = np.arange(0, N, fs/)

fig, (ax, bx) = plt.subplots(2)
ax.plot(a, a_x)
bx.plot(fft_t, fft_ax.real)
plt.show()

# %%

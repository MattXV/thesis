#%%
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy import signal
from scipy.interpolate import interp1d

plt.rcParams["text.usetex"] = True
plt.style.use('science')
#%%
def generate_sine_wave(freq, amplitude, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t)
    return t, sine_wave

def undersample_signal(signal, original_sample_rate, new_sample_rate):
    # Calculate the ratio of old to new sample rate
    sample_rate_ratio = original_sample_rate / new_sample_rate

    # Use array slicing to undersample the signal
    undersampled_signal = signal[::int(sample_rate_ratio)]
    return undersampled_signal

def rounder(values):
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]
    return np.frompyfunc(f, 1, 1)

# Parameters for the sine wave
frequency = 50  # Frequency of the sine wave in Hz
amplitude = 1.0  # Amplitude of the sine wave
duration = 1.0  # Duration of the sine wave in seconds


# Parameters for sampling
original_sample_rate = 1000  # Original sample rate in Hz
undersampled_sample_rate = 90  # Undersampled sample rate in Hz

cycles_to_display = 5
display_time_original = ((duration / frequency) * original_sample_rate) * 5
display_time_under = ((duration / frequency) * undersampled_sample_rate) * 5

# bit-depth 

# Generate the sine wave signal
t, a = generate_sine_wave(frequency, amplitude, duration, original_sample_rate)

# Undersample the signal
b = undersample_signal(a, original_sample_rate, undersampled_sample_rate)
t_undersampled = np.linspace(0, duration, len(b))
b_inter = interp1d(t_undersampled, b)

# bit depth
discrete_values = np.linspace(-1, 1, 9)
b = rounder(discrete_values)(b)

# Plotting

fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300)

ax.plot(t, a, t_undersampled, b, '--')
ax.plot(t_undersampled, b, 'ro')

# ax.plot(t_undersampled, b)
ax.set_xlabel("Time (s), Sampling Points (red ticks)")
ax.set_ylabel("Amplitude")

ax.set_yticks([], minor=True)
ax.set_xticks(t_undersampled, minor=True)
ax.set_yticks(discrete_values, minor=False)
ax.grid(which='both', axis='y', linestyle='--')
ax.tick_params(axis='x', which='minor', length=6, width=2, colors='r')

# ax.tick_params(axis='x', which='minor', length=6, width=2, colors='r')

# ax.stem(t_undersampled, b, linefmt='ro')
# [0:int(display_time_original)]
ax.set_xlim((0, 0.1))
plt.tight_layout()
plt.show()
# %%
fig.savefig('aliasing.pdf', transparent=True)
# %%

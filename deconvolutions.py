# %%
from pathlib import Path
import matplotlib.pyplot as plt
import scienceplots
import signal_tools as st
from scipy.signal import hilbert
import numpy as np

plt.rcParams["text.usetex"] = True
plt.style.use('science')


FS = 48000

# %%
chirp = Path('D:\gdrive\BCU\Research\AcousticRenderingEvaluation\sweep.wav')
mono_recordings = list(Path('D:\gdrive\BCU\Research\AcousticRenderingEvaluation\mono-exports').glob('*.wav'))
test, fs_t = st.sf.read(str(mono_recordings[0]))
chirp, fs_c = st.sf.read(str(chirp))


# %% Deconvolutions

distance_lookup = {'s1l1': 2.6, 's2l2': 2.6, 's3l3': 2.6,
                   's2l1': 3.6, 's3l2': 3.6, 's2l3': 3.6, 's1l2': 3.6, 
                   's1l3': 5.6, 's3l1':5.6}

out_path = Path(r'D:\gdrive\BCU\Research\AcousticRenderingEvaluation\deconvolutions')

for sr_pair in mono_recordings:
    print(sr_pair.stem, distance_lookup[sr_pair.stem.split('-')[1]])
    test, fs_t = st.sf.read(str(sr_pair))
    print('decolving ', fs_t)
    fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 2.2))
    
    distance = distance_lookup[sr_pair.stem.split('-')[1]]
    deconvolved = st.deconvolve(chirp, test, FS, distance)
    t = np.linspace(0, deconvolved.shape[0] / FS, deconvolved.shape[0])
    ax.plot(t, deconvolved)
    st.plot_spectrogram(deconvolved, FS, bx, scale=10)
    
    # st.sf.write(str(out_path / sr_pair.name), deconvolved, FS)

    plt.show()




# %% RIR metrics
# %% 45dB SNR

real_rirs = list(Path(r'D:\gdrive\BCU\Research\AcousticRenderingEvaluation\deconvolutions').glob('*.wav'))
synthetic_rirs = list(Path(r'C:\Users\matt\git\geometrical-acoustics\GeometricalAcoustics\Assets\Irs\acoustic-rendering-eval').glob('*.wav'))
pairs = list()

pairs = [(i, next(filter(lambda x: x.stem in i.stem, real_rirs))) for i in real_rirs]

#%%

fig, axes = plt.subplots(2, 2,  tight_layout=True, dpi=300, figsize=(6.4, 2.2))
ax, bx = axes[0, :]
cx, dx = axes[1, :]
test, fs = st.sf.read(real_rirs[0])
test_s, test_s_fs = st.sf.read(synthetic_rirs[0])
test_s = st.trim_from_to(test_s, 0, 0.5)

t = np.linspace(0, test.shape[0] / fs, test.shape[0])
# ax.plot(t, test)
a = np.abs(hilbert(test))
print(a.shape)
a = np.convolve(a, np.ones(5001)/5001, mode='valid')
print(a.shape)
e = 20 * np.log10(a / np.max(a))
t = np.linspace(0, e.shape[0] / fs, e.shape[0])
ax.plot(np.linspace(0, test.shape[0] / fs, test.shape[0]), test)
bx.plot(t, e)
bx.plot([np.argmax(e < -30) / FS], [-30], marker="*", markersize=13)
print()

a = np.abs(hilbert(test_s))
a = np.convolve(a, np.ones(5001)/5001, mode='valid')
e = 20 * np.log10(a / np.max(a))

cx.plot(np.linspace(0, test_s.shape[0] / fs, test_s.shape[0]), test_s)
dx.plot(np.linspace(0, e.shape[0] / fs, e.shape[0]), e)
dx.plot([np.argmax(e < -30) / FS], [-30], marker="*", markersize=13)
# st.plot_spectrogram(test, fs, bx, scale=10)

# rt_60 = st.get_rt60(test, fs)
# d_50 = st.get_d50(test, fs)
# c_50 = st.get_c50(test, fs)

ax.plot()
plt.show()


# %%



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

fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 2.2))

real_rirs = list(Path(r'G:\My Drive\BCU\Research\AcousticRenderingEvaluation\deconvolutions').glob('*.wav'))

test, fs = st.sf.read(real_rirs[0])

t = np.linspace(0, test.shape[0] / fs, test.shape[0])
# ax.plot(t, test)
a = np.abs(hilbert(test))
a = np.convolve(a, np.ones(5001)/5001, mode='valid')
e = st.mag_to_db(a)
t = np.linspace(0, e.shape[0] / fs, e.shape[0])

bx.plot(t, e)
bx.plot([0, t[-1]], [-30, -30])
# st.plot_spectrogram(test, fs, bx, scale=10)

# rt_60 = st.get_rt60(test, fs)
# d_50 = st.get_d50(test, fs)
# c_50 = st.get_c50(test, fs)

lt = 10* np.log10(np.cumsum(a[:np.argmax(a)][::-1]**2) / np.sum(a[:np.argmax(a)] **2) )[::-1]
ax.plot(lt)
plt.show()


# %%



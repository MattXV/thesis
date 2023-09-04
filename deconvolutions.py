# %%
from pathlib import Path
import matplotlib.pyplot as plt
import scienceplots
import signal_tools as st
import numpy as np

plt.rcParams["text.usetex"] = True
plt.style.use('science')


FS = 48000


chirp = Path('D:\gdrive\BCU\Research\AcousticRenderingEvaluation\sweep.wav')
mono_recordings = list(Path('D:\gdrive\BCU\Research\AcousticRenderingEvaluation\mono-exports').glob('*.wav'))
test, fs_t = st.sf.read(str(mono_recordings[0]))
chirp, fs_c = st.sf.read(str(chirp))


# %%

distance_lookup = {'s1l1': 2.6, 's2l2': 2.6, 's3l3': 2.6,
                   's2l1': 3.6, 's3l2':3.6, 's2l3': 3.6, 's1l2': 3.6, 
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
    
    st.sf.write(str(out_path / sr_pair.name), deconvolved, FS)

    plt.show()
    


# %%

fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 2.2))

print(test.shape, fs_t, chirp.shape, fs_c)
test_dec = st.deconvolve(chirp, test, FS)
t_d = np.argmax(test_dec)

test_dec = test_dec[t_d - round(samples):]

t = np.linspace(0, test_dec.shape[0] / FS, test_dec.shape[0])



st.plot_waveform(test, FS, ax)
bx.plot(t, test_dec)

print()

plt.show()


# %%

st.sf.write('test_dec.wav', test_dec, FS)
# %% Test Convolution

drums = st.read_audio(r'C:\Users\matt\git\geometrical-acoustics\scripts\drums.wav', FS)

conv = st.convolve_ir_to_signal(drums, FS, test_dec, FS, FS)
st.sf.write('test_conv.wav', conv, FS)
# %%

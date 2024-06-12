#%%
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import importlib
import signal_tools as st

plt.rcParams["text.usetex"] = True
plt.style.use('science')

title_fontsize = 12
label_fontsize = 10

FS = 44100

#%% 
importlib.reload(st)
N = 2**14

interp = 2**8
fig, axes = plt.subplots(ncols=3, figsize=(8, 4))
fig.set_layout_engine('tight')

drums   = st.read_audio('drums.wav', FS)
singing = st.read_audio('singing.wav', FS)
arp     = st.read_audio('arp.wav', FS)

f = st.design_bpf(300, 4000, FS)
singing = st.fftconvolve(np.squeeze(singing), f.get_kernel(), 'same')

st.plot_spectrum(drums, FS, axes[0], N, interp)
st.plot_spectrum(singing, FS, axes[1], N, interp)
st.plot_spectrum(arp, FS, axes[2], N, interp)

axes[1].set_xlabel('Frequency (Hz)', fontsize=label_fontsize)
axes[0].set_ylabel('Magnitude',      fontsize=label_fontsize)
axes[0].set_title('Drums',           fontsize=title_fontsize)
axes[1].set_title('Singing',         fontsize=title_fontsize)
axes[2].set_title('Arp',             fontsize=title_fontsize)


plt.show()
# %%

fig.savefig('stimuli.pdf')
# %%

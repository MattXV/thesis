# %%
from pathlib import Path
import matplotlib.pyplot as plt
import scienceplots
import signal_tools as st
import csv
import pandas as pd
import numpy as np

plt.rcParams["text.usetex"] = True
plt.style.use('science')


#%%
FS = 48000


brir_name = 'Kiti Byzantine Church 70cm simplification.J01.Wav'
anechoic_name = 'Viola, Bach NN.wav'
auralisation_name = 'ViolaAuralKiti1.Wav'

anechoic     = st.read_audio(anechoic_name, FS)
brir         = st.read_audio_stereo(brir_name, FS)
auralisation = st.read_audio_stereo(auralisation_name, FS)

print(anechoic.shape)
print(auralisation.shape)
print(brir.shape)

# %%

title_fontsize = 10
label_fontsize = 8

brir = st.trim_from_to(brir, 0, 2, FS)

fig, axes = plt.subplots(5, 2, figsize=(8, 6))
fig.set_layout_engine('tight')

st.plot_waveform(brir[:, 0], FS, axes[0, 0])
st.plot_spectrogram(brir[:, 0], FS, axes[0, 1], scale=100)
axes[0, 0].set_title('BRIR -- Left', fontsize=title_fontsize)
axes[0, 1].set_title('BRIR -- Left', fontsize=title_fontsize)

st.plot_waveform(brir[:, 1], FS, axes[1, 0])
st.plot_spectrogram(brir[:, 1], FS, axes[1, 1], scale=100)
axes[1, 0].set_title('BRIR -- Right', fontsize=title_fontsize)
axes[1, 1].set_title('BRIR -- Right', fontsize=title_fontsize)

st.plot_waveform(anechoic, FS, axes[2, 0])
st.plot_spectrogram(anechoic, FS, axes[2, 1], scale=10)
axes[2, 0].set_title('Anechoic Signal', fontsize=title_fontsize)
axes[2, 1].set_title('Anechoic Signal', fontsize=title_fontsize)

st.plot_waveform(auralisation[:, 0], FS, axes[3, 0])
st.plot_spectrogram(auralisation[:, 0], FS, axes[3, 1], scale=10)
axes[3, 0].set_title('Auralisation -- Left', fontsize=title_fontsize)
axes[3, 1].set_title('Auralisation -- Left', fontsize=title_fontsize)

st.plot_waveform(auralisation[:, 1], FS, axes[4, 0])
st.plot_spectrogram(auralisation[:, 1], FS, axes[4, 1], scale=10)
axes[4, 0].set_title('Auralisation -- Right', fontsize=title_fontsize)
axes[4, 1].set_title('Auralisation -- Right', fontsize=title_fontsize)

axes[4, 0].set_xlabel('Time (s)', fontsize=label_fontsize)
axes[4, 1].set_xlabel('Time (s)', fontsize=label_fontsize)
axes[2, 0].set_ylabel('Magnitude', fontsize=label_fontsize)
axes[2, 1].set_ylabel('Frequency (Hz)', fontsize=label_fontsize)

plt.show()

# %%

fig.savefig('kiti-rir-plots.png', dpi=300)
fig.savefig('kiti-rir-plots.jpg', dpi=300)
# %%

fig.savefig('kiti-rir-plots_uc.pdf')
# %%

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

FS = 48000

#%% Book Chapter


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
#%% 

# *******************************************************************************
# ************************************* Ch 4 Rirs *******************************
# *******************************************************************************

# %% Ch4 Rirs

gt_name      = 'ch4-rirs/gt_dc.wav'
generic_name = 'ch4-rirs/generic-dc.wav'
predicted_name = 'ch4-rirs/tagged-dc.wav'
tagged_name = 'ch4-rirs/predicted-dc.wav'

gt        = st.normalise(np.abs(st.trim_from_to(st.read_audio(gt_name, FS), 0.1, 0.8, FS)))
predicted = st.normalise(np.abs(st.trim_from_to(st.read_audio(predicted_name, FS), 0.11, 0.8, FS)))
tagged    = st.normalise(np.abs(st.trim_from_to(st.read_audio(tagged_name, FS), 0.1, 0.8, FS)))
generic   = st.normalise(np.abs(st.trim_from_to(st.read_audio(generic_name, FS), 0.09, 0.8, FS)))


# %% Ch 4 Rirs

title_fontsize = 12
label_fontsize = 10

fig, axes = plt.subplots(4, figsize=(8, 6))
fig.set_layout_engine('tight')

st.plot_waveform(gt, FS, axes[0])
axes[0].set_title('Ground Truth', fontsize=title_fontsize)

st.plot_waveform(predicted, FS, axes[1])
axes[1].set_title('Predicted', fontsize=title_fontsize)
axes[1].set_ylabel('Magnitude', fontsize=label_fontsize)

st.plot_waveform(tagged, FS, axes[2])
axes[2].set_title('Tagged', fontsize=title_fontsize)

st.plot_waveform(generic, FS, axes[3])
axes[3].set_title('Generic', fontsize=title_fontsize)
axes[3].set_xlabel('Time (s)', fontsize=label_fontsize)

plt.show()

# %%

fig.savefig('rir-texture-testing.pdf')

# %%

title_fontsize = 12
label_fontsize = 10

gt_name      = 'ch4-rirs/gt_dc.wav'
generic_name = 'ch4-rirs/generic-dc.wav'
predicted_name = 'ch4-rirs/tagged-dc.wav'
tagged_name = 'ch4-rirs/predicted-dc.wav'

gt        = st.normalise(st.trim_from_to(st.read_audio(gt_name, FS), 0.1, 0.8, FS))
predicted = st.normalise(st.trim_from_to(st.read_audio(predicted_name, FS), 0.11, 0.8, FS))
tagged    = st.normalise(st.trim_from_to(st.read_audio(tagged_name, FS), 0.1, 0.8, FS))
generic   = st.normalise(st.trim_from_to(st.read_audio(generic_name, FS), 0.09, 0.8, FS))



fig, axes = plt.subplots(4, figsize=(8, 6))
fig.set_layout_engine('tight')

st.plot_spectrogram(gt, FS, axes[0], scale=10)
axes[0].set_title('Ground Truth', fontsize=title_fontsize)

st.plot_spectrogram(predicted, FS, axes[1], scale=10)
axes[1].set_title('Predicted', fontsize=title_fontsize)
axes[1].set_ylabel('Frequency (Hz)', fontsize=label_fontsize)

st.plot_spectrogram(tagged, FS, axes[2], scale=10)
axes[2].set_title('Tagged', fontsize=title_fontsize)

st.plot_spectrogram(generic, FS, axes[3], scale=10)
axes[3].set_title('Generic', fontsize=title_fontsize)
axes[3].set_xlabel('Time (s)', fontsize=label_fontsize)

plt.show()


# %%
fig.savefig('rir-texture-testing-spectrograms.pdf')

# %%

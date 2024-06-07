import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import signal_tools as st

plt.rcParams["text.usetex"] = True
plt.style.use('science')

title_fontsize = 12
label_fontsize = 10

FS = 44100
N = 2**10

fig, axes = plt.subplots(1, figsize=(8, 6))
fig.set_layout_engine('tight')

arp     = st.read_audio('arp.wav', FS)
drums   = st.read_audio('drums.wav', FS)
singing = st.read_audio('singing.wav', FS)

st.plot_spectrum(arp, FS, axes, N)
st.plot_spectrum(drums, FS, axes, N)
st.plot_spectrum(singing, FS, axes, N)

plt.show()
#%%
import matplotlib.pyplot as plt
import scienceplots
import signal_tools as st
import numpy as np
import matplotlib

# matplotlib.use('PS')
plt.rcParams["text.usetex"] = True
plt.style.use('science')

FS = 48000
# %%

fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 3))
t1= 0.10

f125 = st.read_audio('rirs/IR_part_F_125.wav', FS)
f250 = st.read_audio('rirs/IR_part_F_250.wav', FS)
f500 = st.read_audio('rirs/IR_part_F_500.wav', FS)
f1000 = st.read_audio('rirs/IR_part_F_1000.wav', FS)
f2000 = st.read_audio('rirs/IR_part_F_2000.wav', FS)
f4000 = st.read_audio('rirs/IR_part_F_4000.wav', FS)
f125 = st.trim_from_to(f125, 0, t1, FS)
f250 = st.trim_from_to(f250, 0, t1, FS)
f500 = st.trim_from_to(f500, 0, t1, FS)
f1000 = st.trim_from_to(f1000, 0, t1, FS)
f2000 = st.trim_from_to(f2000, 0, t1, FS)
f4000 = st.trim_from_to(f4000, 0, t1, FS)

p125 = st.read_audio('rirs/RT_IR_bin_0.wav', FS)
p250 = st.read_audio('rirs/RT_IR_bin_1.wav', FS)
p500 = st.read_audio('rirs/RT_IR_bin_2.wav', FS)
p1000 = st.read_audio('rirs/RT_IR_bin_3.wav', FS)
p2000 = st.read_audio('rirs/RT_IR_bin_4.wav', FS)
p4000 = st.read_audio('rirs/RT_IR_bin_5.wav', FS)
p125 = st.trim_from_to(np.abs(p125), 0, t1, FS)
p250 = st.trim_from_to(np.abs(p250), 0, t1, FS)
p500 = st.trim_from_to(np.abs(p500), 0, t1, FS)
p1000 = st.trim_from_to(np.abs(p1000), 0, t1, FS)
p2000 = st.trim_from_to(np.abs(p2000), 0, t1, FS)
p4000 = st.trim_from_to(np.abs(p4000), 0, t1, FS)

l1 = st.plot_waveform(f125, FS, bx)
l2 = st.plot_waveform(f250, FS, bx)
l3 = st.plot_waveform(f500, FS, bx)
l4 = st.plot_waveform(f1000, FS, bx)
l5 = st.plot_waveform(f2000, FS, bx)
l6 = st.plot_waveform(f4000, FS, bx)
bx.legend([l1, l2, l3, l4, l5, l6],
          ['F125', 'F250', 'F500', 'F1000', 'F2000', 'F4000'], fontsize=8)

l1 = st.plot_waveform(p125, FS, ax)
l2 = st.plot_waveform(p250, FS, ax)
l3 = st.plot_waveform(p500, FS, ax)
l4 = st.plot_waveform(p1000, FS, ax)
l5 = st.plot_waveform(p2000, FS, ax)
l6 = st.plot_waveform(p4000, FS, ax)
ax.legend([l1, l2, l3, l4, l5, l6],
          ['F125', 'F250', 'F500', 'F1000', 'F2000', 'F4000'], fontsize=8)

ax.set_title('Energy Logged', fontsize=12)
bx.set_title('Poisson-Distributed Energy', fontsize=12)



plt.show()
# %%
fig.savefig('rir-processing.pdf')

# %%


fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 3))
t1 = 0.2

rir = st.read_audio('rirs/tagged.wav', FS)
rir = st.trim_from_to(rir, 0, t1, FS)

line = st.plot_waveform(rir, FS, ax, 'Time Domain')
st.plot_spectrogram(rir, FS, bx)

plt.show()
# %%

fig.savefig('rir-rt-final.pdf', transparent=False)
# %%

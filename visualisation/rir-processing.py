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


fig, rows = plt.subplots(6, tight_layout=True, dpi=300, figsize=(3.2, 3))
fig.suptitle('IR Components')
l1 = st.plot_waveform(p125,  FS, rows[0])
rows[0].set_title('125 Hz')
l2 = st.plot_waveform(p250,  FS, rows[1])
rows[1].set_title('250 Hz')
l3 = st.plot_waveform(p500,  FS, rows[2])
rows[2].set_title('500 Hz')
l4 = st.plot_waveform(p1000, FS, rows[3])
rows[3].set_title('1000 Hz')
l5 = st.plot_waveform(p2000, FS, rows[4])
rows[4].set_title('2000 Hz')
l6 = st.plot_waveform(p4000, FS, rows[5])
rows[5].set_title('4000 Hz')

[rows[i].set_xticklabels([]) for i in range(0, 5)]
rows[5].set_xlabel('Time (s)')
rows[2].set_ylabel('Magnitude')
plt.show()

# %%

fig.savefig('ir-components.pdf')

# %

# %%
fig, (ax) = plt.subplots(1,  tight_layout=True, dpi=300, figsize=(3.2, 3))
t1 = 0.1

rir = st.read_audio('rirs/tagged.wav', FS)
rir = st.trim_from_to(rir, 0, t1, FS)

line = st.plot_waveform(rir, FS, ax, 'Summed IR')
ax.set_title('Summed IR')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Magnitude')

plt.show()

fig.savefig('rir-summed.pdf')

#%%

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


lpf = st.design_lpf(250, FS)
bpf250 = st.design_bpf(250, 500, FS)
bpf500 = st.design_bpf(500, 1000, FS)
bpf1000 = st.design_bpf(1000, 2000, FS)
bpf2000 = st.design_bpf(2000, 4000, FS)
hpf = st.design_hpf(4000, FS)
lpf_fft = np.abs(st.rfft(lpf.get_kernel(), st.FILTER_KERNEL_SIZE))
bpf250_fft = np.abs(st.rfft(bpf250.get_kernel(), st.FILTER_KERNEL_SIZE))
bpf500_fft = np.abs(st.rfft(bpf500.get_kernel(), st.FILTER_KERNEL_SIZE))
bpf1000_fft = np.abs(st.rfft(bpf1000.get_kernel(), st.FILTER_KERNEL_SIZE))
bpf2000_fft = np.abs(st.rfft(bpf2000.get_kernel(), st.FILTER_KERNEL_SIZE))
hpf_fft = np.abs(st.rfft(hpf.get_kernel(), st.FILTER_KERNEL_SIZE))


fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 2.8))

t = np.linspace(0, FS/2, lpf_fft.shape[0])

l1, = ax.loglog(t, lpf_fft)
l3, = ax.loglog(t, bpf250_fft)
l4, = ax.loglog(t, bpf500_fft)
l5, = ax.loglog(t, bpf1000_fft)
l6, = ax.loglog(t, bpf2000_fft)
l7, = ax.loglog(t, hpf_fft)


ax.set_xlim(20, 20000)
ax.set_ylim([10e-8, 10])
ax.legend([l1, l3, l4, l5, l6, l7], 
          [r'lpf', r'$\textrm{bpf}_{250}$',
           r'$\textrm{bpf}_{500}$',r'$\textrm{bpf}_{1000}$', r'$\textrm{bpf}_{2000}$',
           r'hpf'], fontsize=8)
ax.set_xlabel('Frequency (Hz)', fontsize=10)
ax.set_ylabel('Attenuation', fontsize=10)
ax.set_title('Filter Bank', fontsize=12)
p125 = lpf.convolve(np.squeeze(p125))
p250 = bpf250.convolve(np.squeeze(p250))
p500 = bpf500.convolve(np.squeeze(p500))
p1000 = bpf1000.convolve(np.squeeze(p1000))
p2000 = bpf2000.convolve(np.squeeze(p2000))
p4000 = hpf.convolve(np.squeeze(p4000))

l1 = st.plot_waveform(p125,  FS, bx)
l2 = st.plot_waveform(p250,  FS, bx)
l3 = st.plot_waveform(p500,  FS, bx)
l4 = st.plot_waveform(p1000, FS, bx)
l5 = st.plot_waveform(p2000, FS, bx)
l6 = st.plot_waveform(p4000, FS, bx, fontsize=10)
bx.legend([l1, l2, l3, l4, l5, l6],
          ['F125', 'F250', 'F500', 'F1000', 'F2000', 'F4000'], fontsize=9)
bx.set_title('Filtered IR Parts', fontsize=12)

plt.show()
# %%
fig.savefig('filterbank.eps', transparent=True)
fig.savefig('filterbank.pdf', transparent=True)

# %%

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import scienceplots
import signal_tools as st

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

pairs = [(i, next(filter(lambda x: x.stem in i.stem, real_rirs))) for i in synthetic_rirs]
pairs_a = list(filter(lambda x: 'a' in x[0].stem, pairs))
pairs_b = list(filter(lambda x: 'b' in x[0].stem, pairs))


#%%
plt.ioff()
fig, axes = plt.subplots(5, 3,  tight_layout=True, dpi=300, figsize=(6.4, 8.2))
fig.suptitle(r'Grid A', fontsize=12)
fontsize_legend = 9
fontsize_title = 12
fontsize_label = 9
ax, bx, cx = axes[0, :]

synthetic_fn, real_fn = pairs_a[0]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)
cx.legend([l1, l2], ['Simulated', 'Real'], fontsize=fontsize_legend)
ax.set_title('Synthetic', fontsize=fontsize_title)
bx.set_title('Measured', fontsize=fontsize_title)
cx.set_title('Decay Curves (dB)', fontsize=fontsize_title)


# %% ROW 2
ax, bx, cx = axes[1, :]
synthetic_fn, real_fn = pairs_a[1]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%
ax, bx, cx = axes[2, :]
synthetic_fn, real_fn = pairs_a[2]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)
# %%


ax, bx, cx = axes[3, :]
synthetic_fn, real_fn = pairs_a[3]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)
# %%

ax, bx, cx = axes[4, :]
synthetic_fn, real_fn = pairs_a[4]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%

# ax, bx, cx = axes[5, :]
# synthetic, real = pairs[5]

# synthetic, synthetic_fs = st.sf.read(synthetic)
# synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
# real, real_fs = st.sf.read(real)
# real = st.normalise(real)
# assert(real_fs == synthetic_fs == FS)
# ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
# bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

# decay_st, decay_s = st.get_decay_curve(synthetic, FS)
# decay_rt, decay_r = st.get_decay_curve(real, FS)
# l1, = cx.plot(decay_st, decay_s)
# l2, = cx.plot(decay_rt, decay_r)
# cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
# cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%

plt.show()

# %%

fig.savefig('real_synthetic_rir.pdf', transparent=True)

# %% ROW 6


plt.ioff()
fig, axes = plt.subplots(4, 3,  tight_layout=True, dpi=300, figsize=(6.4, 8.2))
fig.suptitle(r'Grid A', fontsize=12)

ax, bx, cx = axes[0, :]
synthetic_fn, real_fn = pairs_a[5]

row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)
cx.legend([l1, l2], ['Simulated', 'Real'], fontsize=fontsize_legend)
ax.set_title('Synthetic', fontsize=fontsize_title)
bx.set_title('Measured', fontsize=fontsize_title)
cx.set_title('Decay Curves (dB)', fontsize=fontsize_title)

#%%
ax, bx, cx = axes[1, :]
synthetic_fn, real_fn = pairs_a[6]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%

#%%
ax, bx, cx = axes[2, :]
synthetic_fn, real_fn = pairs_a[7]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%

ax, bx, cx = axes[3, :]
synthetic_fn, real_fn = pairs_a[8]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%
plt.show()

# %%
fig.savefig('real_synthetic_rir_cont.pdf', transparent=True)
# %%
#
#
#
#       grid b
#
#
#
#
#
#
#
#
#
#
#%%
plt.ioff()
fig, axes = plt.subplots(5, 3,  tight_layout=True, dpi=300, figsize=(6.4, 8.2))
fig.suptitle(r'Grid B', fontsize=12)
fontsize_legend = 9
fontsize_title = 12
fontsize_label = 9
ax, bx, cx = axes[0, :]

synthetic_fn, real_fn = pairs_b[0]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)
cx.legend([l1, l2], ['Simulated', 'Real'], fontsize=fontsize_legend)
ax.set_title('Synthetic', fontsize=fontsize_title)
bx.set_title('Measured', fontsize=fontsize_title)
cx.set_title('Decay Curves (dB)', fontsize=fontsize_title)


# %% ROW 2
ax, bx, cx = axes[1, :]
synthetic_fn, real_fn = pairs_b[1]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%
ax, bx, cx = axes[2, :]
synthetic_fn, real_fn = pairs_b[2]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)
# %%


ax, bx, cx = axes[3, :]
synthetic_fn, real_fn = pairs_b[3]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)
# %%

ax, bx, cx = axes[4, :]
synthetic_fn, real_fn = pairs_b[4]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%

# ax, bx, cx = axes[5, :]
# synthetic, real = pairs[5]

# synthetic, synthetic_fs = st.sf.read(synthetic)
# synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
# real, real_fs = st.sf.read(real)
# real = st.normalise(real)
# assert(real_fs == synthetic_fs == FS)
# ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
# bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

# decay_st, decay_s = st.get_decay_curve(synthetic, FS)
# decay_rt, decay_r = st.get_decay_curve(real, FS)
# l1, = cx.plot(decay_st, decay_s)
# l2, = cx.plot(decay_rt, decay_r)
# cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
# cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%

plt.show()

# %%

fig.savefig('real_synthetic_rir_b.pdf', transparent=True)

# %% ROW 6


plt.ioff()
fig, axes = plt.subplots(4, 3,  tight_layout=True, dpi=300, figsize=(6.4, 8.2))
fig.suptitle(r'Grid B', fontsize=12)

ax, bx, cx = axes[0, :]
synthetic_fn, real_fn = pairs_b[5]

row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)
cx.legend([l1, l2], ['Simulated', 'Real'], fontsize=fontsize_legend)
ax.set_title('Synthetic', fontsize=fontsize_title)
bx.set_title('Measured', fontsize=fontsize_title)
cx.set_title('Decay Curves (dB)', fontsize=fontsize_title)

#%%
ax, bx, cx = axes[1, :]
synthetic_fn, real_fn = pairs_b[6]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%

#%%
ax, bx, cx = axes[2, :]
synthetic_fn, real_fn = pairs_b[7]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%

ax, bx, cx = axes[3, :]
synthetic_fn, real_fn = pairs_b[8]
row_name = synthetic_fn.stem.split('-')[-1].upper()
ax.set_ylabel('{} {}'.format(row_name[:2], row_name[2:]), fontsize=fontsize_label)

synthetic, synthetic_fs = st.sf.read(synthetic_fn)
synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
real, real_fs = st.sf.read(real_fn)
real = st.normalise(real)
assert(real_fs == synthetic_fs == FS)
ax.plot(np.linspace(0, synthetic.shape[0] / FS, synthetic.shape[0]), synthetic)
bx.plot(np.linspace(0, real.shape[0] / FS, real.shape[0]), real)

decay_st, decay_s = st.get_decay_curve(synthetic, FS)
decay_rt, decay_r = st.get_decay_curve(real, FS)
l1, = cx.plot(decay_st, decay_s)
l2, = cx.plot(decay_rt, decay_r)
cx.plot([np.argmax(decay_s < -30) / FS], [-30], marker='*', markersize=12)
cx.plot([np.argmax(decay_r < -30) / FS], [-30], marker='*', markersize=12)

# %%
plt.show()

# %%
fig.savefig('real_synthetic_rir_bcont.pdf', transparent=True)
# %%

header = ['Position Pair', 'T\textsubscript{30} Simulated', 'T\textsubscript{30} Real',
          'C\textsubscript{50} Simulated', 'C\textsubscript{50} Real', 'D\textsubscript{50} Simulated', 'D\textsubscript{50} Real']
metrics_a = [header]
metrics_b = [header]
for pair_a, pair_b in zip(pairs_a, pairs_b):
    for pair in [pair_a, pair_b]:
        synthetic_fn, real_fn = pair
        row_name = synthetic_fn.stem.split('-')[-1].upper()
        grid = synthetic_fn.stem.split('-')[0]

        synthetic, synthetic_fs = st.sf.read(synthetic_fn)
        synthetic = st.trim_from_to(st.normalise(synthetic), 0, 0.3)
        real, real_fs = st.sf.read(real_fn)
        real = st.normalise(real)

        t, a = st.get_decay_curve(synthetic, FS)
        t_30_synthetic = np.argmax(a < -30) / FS
        t, a = st.get_decay_curve(real, FS)
        t_30_real = np.argmax(a < -30) / FS

        c50_synthetic = abs(st.get_c50(synthetic, FS))
        c50_real = abs(st.get_c50(real, FS))

        d50_synthetic = st.get_d50(synthetic, FS)
        d50_real = st.get_d50(real, FS)

        row = ['{} {}'.format(row_name[:2], row_name[2:]), 
               '{:3.2f}'.format(t_30_synthetic), '{:3.2f}'.format(t_30_real),
               '{:3.2f}'.format(c50_synthetic), '{:3.2f}'.format(c50_real),
               '{:3.2f}'.format(d50_synthetic), '{:3.2f}'.format(d50_real)]

        if 'a' in grid:
            metrics_a.append(row)
        else:
            metrics_b.append(row)




# %%

import csv
import pandas as pd
# %%

with open('metrics_grid_a.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(metrics_a[0])
    writer.writerows(metrics_a[1:])



with open('metrics_grid_b.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(metrics_b[0])
    writer.writerows(metrics_b[1:])
# %%

grid_a = pd.read_csv('metrics_grid_a.csv')
grid_b = pd.read_csv('metrics_grid_b.csv')

# %%
fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 2.2))
x = [0, 1, 2]
lines = list()

l1 = grid_a[grid_a['Position Pair'].str.contains('L1')]
l2 = grid_a[grid_a['Position Pair'].str.contains('L2')]
l3 = grid_a[grid_a['Position Pair'].str.contains('L3')]
y_sim = list(l1['C\textsubscript{50} Simulated'])
y_real = list(l1['C\textsubscript{50} Real'])
lines.extend(ax.plot(x, y_sim, color='tab:blue'))
lines.extend(ax.plot(x, y_real, color='tab:blue', linestyle='dashed'))
y_sim = list(l2['C\textsubscript{50} Simulated'])
y_real = list(l2['C\textsubscript{50} Real'])
lines.extend(ax.plot(x, y_sim,  color='tab:red'))
lines.extend(ax.plot(x, y_real, color='tab:red', linestyle='dashed'))
y_sim = list(l3['C\textsubscript{50} Simulated'])
y_real = list(l3['C\textsubscript{50} Real'])
lines.extend(ax.plot(x, y_sim,  color='tab:green'))
lines.extend(ax.plot(x, y_real, color='tab:green', linestyle='dashed'))

ax.legend(lines, ['L1 Sim', 'L1 Real', 'L2 Sim', 'L2 Real', 'L3 Sim', 'L3 Real'],
          fontsize=fontsize_legend)
ax.set_ylabel(r'C\textsubscript{50} (dB)', fontsize=fontsize_label)
ax.set_xticks(x, ['S1', 'S2', 'S3'], fontsize=fontsize_label)
ax.set_title('Grid A', fontsize=fontsize_title)

l1 = grid_b[grid_b['Position Pair'].str.contains('L1')]
l2 = grid_b[grid_b['Position Pair'].str.contains('L2')]
l3 = grid_b[grid_b['Position Pair'].str.contains('L3')]
y_sim = list(l1['C\textsubscript{50} Simulated'])
y_real = list(l1['C\textsubscript{50} Real'])
lines.extend(bx.plot(x, y_sim, color='tab:blue'))
lines.extend(bx.plot(x, y_real, color='tab:blue', linestyle='dashed'))
y_sim = list(l2['C\textsubscript{50} Simulated'])
y_real = list(l2['C\textsubscript{50} Real'])
lines.extend(bx.plot(x, y_sim,  color='tab:red'))
lines.extend(bx.plot(x, y_real, color='tab:red', linestyle='dashed'))
y_sim = list(l3['C\textsubscript{50} Simulated'])
y_real = list(l3['C\textsubscript{50} Real'])
lines.extend(bx.plot(x, y_sim,  color='tab:green'))
lines.extend(bx.plot(x, y_real, color='tab:green', linestyle='dashed'))

bx.legend(lines, ['L1 Sim', 'L1 Real', 'L2 Sim', 'L2 Real', 'L3 Sim', 'L3 Real'],
          fontsize=fontsize_legend)
bx.set_ylabel(r'C\textsubscript{50} (dB)', fontsize=fontsize_label)
bx.set_xticks(x, ['S1', 'S2', 'S3'], fontsize=fontsize_label)
bx.set_title('Grid B', fontsize=fontsize_title)


plt.show()

# %%
fig.savefig('c50.pdf', transparent=True)

# %%
fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 2.2))
x = [0, 1, 2]
lines = list()

l1 = grid_a[grid_a['Position Pair'].str.contains('L1')]
l2 = grid_a[grid_a['Position Pair'].str.contains('L2')]
l3 = grid_a[grid_a['Position Pair'].str.contains('L3')]
y_sim = list(l1['D\textsubscript{50} Simulated'])
y_real = list(l1['D\textsubscript{50} Real'])
lines.extend(ax.plot(x, y_sim, color='tab:blue'))
lines.extend(ax.plot(x, y_real, color='tab:blue', linestyle='dashed'))
y_sim = list(l2['D\textsubscript{50} Simulated'])
y_real = list(l2['D\textsubscript{50} Real'])
lines.extend(ax.plot(x, y_sim,  color='tab:red'))
lines.extend(ax.plot(x, y_real, color='tab:red', linestyle='dashed'))
y_sim = list(l3['D\textsubscript{50} Simulated'])
y_real = list(l3['D\textsubscript{50} Real'])
lines.extend(ax.plot(x, y_sim,  color='tab:green'))
lines.extend(ax.plot(x, y_real, color='tab:green', linestyle='dashed'))

ax.legend(lines, ['L1 Sim', 'L1 Real', 'L2 Sim', 'L2 Real', 'L3 Sim', 'L3 Real'],
          fontsize=fontsize_legend)
ax.set_ylabel(r'D\textsubscript{50} (dB)', fontsize=fontsize_label)
ax.set_xticks(x, ['S1', 'S2', 'S3'], fontsize=fontsize_label)
ax.set_title('Grid A', fontsize=fontsize_title)

l1 = grid_b[grid_b['Position Pair'].str.contains('L1')]
l2 = grid_b[grid_b['Position Pair'].str.contains('L2')]
l3 = grid_b[grid_b['Position Pair'].str.contains('L3')]
y_sim = list(l1['D\textsubscript{50} Simulated'])
y_real = list(l1['D\textsubscript{50} Real'])
lines.extend(bx.plot(x, y_sim, color='tab:blue'))
lines.extend(bx.plot(x, y_real, color='tab:blue', linestyle='dashed'))
y_sim = list(l2['D\textsubscript{50} Simulated'])
y_real = list(l2['D\textsubscript{50} Real'])
lines.extend(bx.plot(x, y_sim,  color='tab:red'))
lines.extend(bx.plot(x, y_real, color='tab:red', linestyle='dashed'))
y_sim = list(l3['D\textsubscript{50} Simulated'])
y_real = list(l3['D\textsubscript{50} Real'])
lines.extend(bx.plot(x, y_sim,  color='tab:green'))
lines.extend(bx.plot(x, y_real, color='tab:green', linestyle='dashed'))

bx.legend(lines, ['L1 Sim', 'L1 Real', 'L2 Sim', 'L2 Real', 'L3 Sim', 'L3 Real'],
          fontsize=fontsize_legend)
bx.set_ylabel(r'D\textsubscript{50} (dB)', fontsize=fontsize_label)
bx.set_xticks(x, ['S1', 'S2', 'S3'], fontsize=fontsize_label)
bx.set_title('Grid B', fontsize=fontsize_title)


plt.show()

# %%
fig.savefig('D50.pdf', transparent=True)

# %%

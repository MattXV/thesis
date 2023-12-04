from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy import interpolate
from scipy.signal import find_peaks, resample, spectrogram, fftconvolve, hilbert
from scipy.fft import fft, ifft, rfft, irfft
from scipy.signal.windows import blackman
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


OUT = 'dbscale/'
plt.style.use('science')
plt.rcParams['font.size'] = '5'


SAMPLERATE = 48000
# SPECTROGRAM_NSEGS = 2**9
SPECTROGRAM_NFFT = 2**9
SPECTROGRAM_WSIZE = 64
SPECTROGRAM_NSEGS = 2**10
# SPECTROGRAM_NFFT = 1024
FILTER_KERNEL_SIZE = 2**14


def clip(signal):
    return np.where(signal > 0.2, 0.9, signal)


def normalise(signal):
    return signal / np.abs(np.max(signal))


def interp_signal(signal, points):
    interp = interpolate.interp1d(np.arange(signal.shape[0]), np.squeeze(signal))
    x = np.linspace(0, signal.shape[0] - 1, points)
    return x, interp(x)


def mag_to_db(signal, threshold=-80):
    input_signal = np.abs(np.squeeze(np.copy(signal)))
    thresh = db_to_mag(threshold)
    x = np.ones(max(input_signal.shape), np.float32) * thresh
    x = np.where(input_signal > thresh, input_signal, x)
    return 20 * np.log10(x)


def db_to_mag(db):
    return 10 ** (np.copy(db) / 20)

def get_decay_curve(rir, fs, smoothing=5001):
    a = np.abs(hilbert(rir))
    a = np.convolve(a, np.ones(smoothing)/smoothing, mode='valid')
    return np.linspace(0, a.shape[0] / fs, a.shape[0]), 20 * np.log10(a / np.max(a))

def get_rir_segments(rir, fs):
    x = mag_to_db(np.copy(rir) ** 2)
    # direct sound
    peaks, _ = find_peaks(x, prominence=db_to_mag(-30), distance=int(fs * 0.1 * 10e-3))
    max_peak = np.argsort(x[peaks])[::-1]
    td = peaks[max_peak[0]]
    # Segment power function. De Lima et al.
    direct = x[:int(td - 1 * 10e-3 * fs)]
    ed = x[int(td - (1 * 10e-3) * fs):int(td + (1.5 * 10e-3) * fs)]
    er = x[int(td + (1.5 * 10e-3) * fs):]
    return x, td, ed, er, peaks


def get_rt60(signal, samplerate):
    db_level = db_to_mag(-60)
    _, td, _, _, _ = get_rir_segments(signal, samplerate)
    decay = np.squeeze(mag_to_db(signal[td:]))

    rt_60 = list()
    for value in decay:
        if abs(value) > -60:
            rt_60.append(value)
        else:
            break
    return len(rt_60) / samplerate


def get_c50(signal, samplerate):
    x = np.squeeze(np.copy(signal))
    t_x = np.max(x.shape) / samplerate
    h = lambda t : abs(x[int(t * samplerate)])
    nom, _ = quad(h, 0, 0.05, limit=50000)
    den, _ = quad(h, 0.05, t_x, limit=50000)
    return 10 * np.log10(nom / den)


def get_d50(signal, samplerate):
    x = np.squeeze(np.copy(signal))
    t_x = np.max(x.shape) / samplerate
    h = lambda t : abs(x[int(t * samplerate)])
    nom, _ = quad(h, 0, 0.05, limit=50000)
    den, _ = quad(h, 0, t_x, limit=50000)
    return nom / den


def get_sound_strength(near_rir, far_rir, samplerate):
    t_near = near_rir.shape[0] / samplerate
    t_far = far_rir.shape[0] / samplerate
    h_near = lambda t : abs(near_rir[int(t * samplerate)])**2
    h_far = lambda t: abs(far_rir[int(t * samplerate)])**2
    nom, _ = quad(h_near, 0, t_near, limit=1000000)
    den, _ = quad(h_far, 0, t_far, limit=1000000)
    return 10 * np.log10(nom / den)


def read_audio(file, out_samplerate):
    x, fs = sf.read(file, always_2d=True)
    if (x.shape[1] == 2):
        x = x[:, 0] + x[:, 1]
    x = normalise(x)
    t = np.max(x.shape) / fs
    x = resample(x, int(np.floor(t * fs)))
    x = normalise(x)
    return x

def read_audio_stereo(file, out_samplerate):
    x, fs = sf.read(file, always_2d=True)
    x = normalise(x)
    t = np.max(x.shape) / fs
    x = resample(x, int(np.floor(t * fs)))
    x = normalise(x)
    return x


def trim_from_to(signal, time_a, time_b, fs=None):
    fs = SAMPLERATE if fs is None else fs
    start = int(time_a * fs)
    stop = int(time_b * fs)
    y = signal.copy()[start:stop]
    return y


def convolve_ir_to_signal(signal, signal_samplerate, ir, ir_samplerate, out_samplerate):
    if not any([isinstance(signal, np.ndarray), isinstance(signal, str)]):
        raise TypeError('Signal must be either an np.array or a path to an \
                        audio file')
    if not any([isinstance(ir, np.ndarray), isinstance(ir, str)]):
        raise TypeError('Ir must be either an np.array or a path to an \
                        audio file')
    f = signal.copy()
    g = ir.copy()
    if len(f.shape) > 1:
        if f.shape[1] == 2:
            f = f[:, 0] + f[:, 1]
        elif f.shape[1] == 1:
            pass
        else:
            raise TypeError('Signal has incorrect number of channels: only \
                             stereo or mono are supported!')
    if len(g.shape) > 1:
        if g.shape[1] == 2:
            g = g[:, 0] + g[:, 1]
        elif g.shape[1] == 1:
            pass
        else:
            raise TypeError('RIR has incorrect number of channels: only \
                             stereo or mono are supported!')

    t = np.max(f.shape) / ir_samplerate
    f = resample(f, int(np.floor(t * out_samplerate)))
    f = normalise(f)
    t = np.max(g.shape) / signal_samplerate
    g = resample(g, int(np.floor(t * out_samplerate)))
    g = normalise(g)
    y = np.convolve(np.squeeze(g), np.squeeze(f), 'full')
    y = normalise(y)
    return y

def deconvolve(measurement_chirp, recording, samplerate, source_listener_dist_m=None):
    measurement_chirp = np.squeeze(measurement_chirp)
    recording = np.squeeze(recording)
    bpf = design_bpf(25, 20000, samplerate)
    assert(len(measurement_chirp.shape) == 1 and len(recording.shape) == 1)
    n_conv = measurement_chirp.shape[0] + recording.shape[0]
    x_fft = rfft(measurement_chirp[::-1], n_conv, norm='ortho')
    y_fft = rfft(recording, n_conv, norm='ortho')
    filter_fft = rfft(bpf.get_kernel(), n_conv, norm='ortho')
    z_fft = x_fft * y_fft * filter_fft
    z = irfft(z_fft, n_conv, norm='ortho')
    z = z[measurement_chirp.shape[0]:]
    z = z[:z.shape[0] - measurement_chirp.shape[0]]
    z = z / np.max(np.abs(z))

    # time shifting correction
    if source_listener_dist_m is None: return z
    delay_samples = round((source_listener_dist_m / 343) * samplerate)
    z = z[np.argmax(z) - delay_samples:]

    return z

# PLOTS
def plot_spectrogram(signal, samplerate, axes, label=None, scale=1000, title_fontsize=18, fontsize=12, labels=('Frequency (Hz)', 'Time (s)')):
    # Pxx, freqs, bins, im = axes.specgram(signal[:, 0],
    #     Fs=samplerate, sides='onesided', mode='psd',
    #     pad_to=SPECTROGRAM_NSEGS, NFFT=SPECTROGRAM_NFFT,
    #     scale='dB', cmap='coolwarm', scale_by_freq=False)
    y = np.copy(signal)
    y = y * scale
    y = np.squeeze(y)
    y = np.where(y >= 1, 0.9, y)
    y = np.where(y <= -1, -0.9, y)
    f, t, Sxx = spectrogram(np.squeeze(y), samplerate, window=blackman(SPECTROGRAM_WSIZE),
                            nfft=SPECTROGRAM_NFFT, mode='magnitude')
    norm = Normalize(clip=True)
    axes.pcolormesh(t, f, Sxx, cmap='gist_heat', norm=norm, shading='auto')
    # if label:
    #     axes.set_title(label, fontsize=title_fontsize)
    # if labels:
    #     axes.set_ylabel(labels[0], fontsize=fontsize)
    #     axes.set_xlabel(labels[1], fontsize=fontsize)
    axes.set_yscale('log')
    axes.set_ylim(20, 20000)


def plot_db_scale(signal, samplerate, axes, label='Signal'):
    fig, ax = plt.subplots()
    db_scale = mag_to_db(signal)
    # db_scale = normalise(db_scale)
    db_scale = np.abs(db_scale)
    db_scale = np.squeeze(np.array(np.where(db_scale > -60)))
    t = np.max(db_scale.shape) / samplerate
    # axes.plot(np.linspace(0, t, np.max(db_scale.shape)), db_scale)
    axes.plot(signal)
    axes.set_yticks([])
    axes.set_xticks([])

    # fig.savefig(str(Path(OUT) / '{}.png'.format(label)))
    plt.close(fig)
    del fig



def plot_waveform(signal, samplerate, axes, label=None, td=None, title_fontsize=15, fontsize=12, labels=('Time (s)', 'Amplitude')):
    # if not td:
    #     _, td, _, _, _ = get_rir_segments(signal, samplerate)
    # signal = signal[td - int(0.002 * samplerate):]

    t = np.max(signal.shape) / samplerate
    points = 150


    t_db, x = interp_signal(signal, points)
    db_scale = mag_to_db(x)

    line, = axes.plot(np.linspace(0, t, signal.shape[0]), signal)
    # axes.plot(np.linspace(0, t, t_db.shape[0]), db_scale)
    
    # axes.plot(t_db, db_scale)

    return line


def plot_image(image, axes, label=None, title=None):
    if label:
        axes.set_ylabel(label, fontsize=18, rotation='vertical')
    if title:
        axes.set_title(title, fontsize=18)
    axes.imshow(image)
    axes.set_xticks([])
    axes.set_yticks([])

def real_fft(x):
    z = fft(x)
    z = np.abs(x[:x.shape[0] // 2])
    return z


class Filter:
    def __init__(self):
        self.kernel = np.zeros(FILTER_KERNEL_SIZE, np.float32)
    
    def convolve(self, x):
        return fftconvolve(x, self.kernel, 'same')

    def get_kernel(self): 
        return self.kernel


def design_lpf(fc, fs):
    fc = fc / fs
    f = Filter()
    m = f.kernel.shape[0]
    f_sum = 0
    for i in range(m):
        n = i - m / 2
        if n == 0:
            f.kernel[i] = 2 * math.pi * fc
        else:
            f.kernel[i] = math.sin(2 * math.pi * fc * n) / n
        f.kernel[i] = f.kernel[i] * (0.42 - 0.5 * math.cos(2 * math.pi * i / m) + 0.08 * math.cos(4 * math.pi * i / m))
        f_sum += f.kernel[i]
    f.kernel /= f_sum
    return f
    

def design_hpf(fc, fs):
    f = design_lpf(fc, fs)
    m = f.kernel.shape[0]
    f.kernel = -f.kernel
    f.kernel[m // 2] += 1
    return f


def design_bpf(lower_fc, upper_fc, fs):
    f = design_lpf(upper_fc, fs)
    hpf = design_hpf(lower_fc, fs)
    m = f.kernel.shape[0]
    for i in range(m):
        f.kernel[i] = -(f.kernel[i] + hpf.kernel[i])
        if i == m // 2:
            f.kernel[i] += 1
    return f
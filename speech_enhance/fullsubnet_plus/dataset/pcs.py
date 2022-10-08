import os
import torch
import torchaudio
import numpy as np
import argparse
import librosa
import scipy

PCS = np.ones(257)      # Perceptual Contrast Stretching
PCS[0:3] = 1
PCS[3:6] = 1.070175439
PCS[6:9] = 1.182456140
PCS[9:12] = 1.287719298
PCS[12:138] = 1.4       # Pre Set
PCS[138:166] = 1.322807018
PCS[166:200] = 1.238596491
PCS[200:241] = 1.161403509
PCS[241:256] = 1.077192982

maxv = np.iinfo(np.int16).max


def Sp_and_phase(signal):
    signal_length = signal.shape[0]
    n_fft = 512
    y_pad = librosa.util.fix_length(signal, signal_length + n_fft // 2)

    F = librosa.stft(y_pad, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)

    Lp = PCS * np.transpose(np.log1p(np.abs(F)), (1, 0))
    phase = np.angle(F)

    NLp = np.transpose(Lp, (1, 0))

    return NLp, phase, signal_length


def SP_to_wav(mag, phase, signal_length):
    mag = np.expm1(mag)
    Rec = np.multiply(mag, np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming, length=signal_length)
    return result


def PCS_Aug(wav):
    noisy_LP, Nphase, signal_length = Sp_and_phase(wav.squeeze().numpy())

    enhanced_wav = SP_to_wav(noisy_LP, Nphase, signal_length)
    enhanced_wav = enhanced_wav/np.max(abs(enhanced_wav))

    return torch.from_numpy(enhanced_wav).type(torch.float32)

def PCS_Aug_np(wav):
    noisy_LP, Nphase, signal_length = Sp_and_phase(wav)

    enhanced_wav = SP_to_wav(noisy_LP, Nphase, signal_length)
    enhanced_wav = enhanced_wav/np.max(abs(enhanced_wav))

    return enhanced_wav
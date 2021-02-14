import dienen
import datasets
import generators
from IPython import embed
from callbacks import WANDBLogger
import wandb
import tensorflow as tf
from swissknife.dsp import istft, stft, calculate_synthesis_window,pghi, get_default_window
import librosa
import numpy as np
import soundfile as sf

ae_model = dienen.Model('models/gsvqvae.yaml')
ae_model.build()
ae_model.core_model.model.summary()
ae_model.core_model.model.load_weights('../ckpts_gsvqvae/weights.07-155.30.hdf5')

input_frames = 16
window_size = 1024
hop_size = 256

x, fs = librosa.core.load('../nsynth-valid/audio/guitar_acoustic_010-079-025.wav')
X = stft(x,window_size,hop_size,window=get_default_window(window_size)[0])
X = X[:,:-1]

X_frames = np.array([X[i:i+input_frames] for i in range(0,len(X)-input_frames,input_frames)])
X_frames = np.log(X_frames+1e-16)
Y_frames = ae_model.core_model.model.predict(X_frames)
Y_frames = np.squeeze(Y_frames)
Y_frames = np.pad(Y_frames,((0,0),(0,0),(0,1))) + 1e-16
Y_frames = np.exp(Y_frames)
Y = np.concatenate(Y_frames)

synth_window = calculate_synthesis_window(win_length=window_size, hop_length=hop_size, n_fft=window_size,window=get_default_window(window_size)[0])
Y = np.abs(Y)*np.exp(1.0j*pghi(Y,window_size,hop_size,synthesis_window=synth_window))
y = istft(Y,window_size,hop_size,synthesis_window=synth_window)

sf.write('test.wav', y, fs)

embed()
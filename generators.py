import librosa
import pandas as pd
from pathlib import Path
import numpy as np
import tqdm
import soundfile as sf
from tensorflow.keras.utils import Sequence
from IPython import embed
from swissknife.dsp import stft, get_default_window

class PGHIGenerator(Sequence):
    def __init__(self,audio_metadata,batch_size=16,frame_size=256,hop_size=64,crop_nyquist=True,debug_times=False,shape_batches=None,apply_log=False,normalize=False):
        self.data = audio_metadata
        self.index = np.array(self.data.index)
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window = get_default_window(self.frame_size)[0]
        self.crop_nyquist = crop_nyquist
        self.debug_times = debug_times
        self.shape_batches = shape_batches
        self.apply_log = apply_log
        self.normalize = normalize
        if self.normalize and not isinstance(self.normalize,dict):
            self.normalize = {'mean': 0, 'std': 1, 'minus': None, 'multiplier': None}

    def read_wav(self,x):
        try:
            return sf.read(x['file_path'],start=x['start'],stop=x['end'])[0]
        except:
            return np.zeros((x['end']-x['start']))

    def __getitem__(self,i):
        if not self.debug_times:
            batch_idxs = np.take(self.index,np.arange(i*self.batch_size,(i+1)*self.batch_size),mode='wrap')
            batch_df = self.data.loc[batch_idxs]
            
            batch_x = batch_df.apply(lambda x: self.read_wav(x),axis=1)
            batch_x = np.stack(batch_x)
            batch_x = np.array([stft(x_i,self.frame_size,self.hop_size,window=self.window) for x_i in batch_x])
            batch_x = np.abs(batch_x)
            if self.apply_log:
                batch_x = np.log(batch_x+1e-16)
            if self.normalize:
                if self.normalize.get('minus',None) is not None:
                    batch_x -= self.normalize['minus']
                else:
                    batch_x -= np.mean(batch_x)
                if self.normalize.get('multiplier',None) is not None:
                    batch_x *= self.normalize['multiplier']
                else:
                    batch_x /= np.std(batch_x)
                batch_x += self.normalize.get('mean',0)
                batch_x *= self.normalize.get('std',1)

            if self.crop_nyquist:
                batch_x = batch_x[:,:,:-1]
        else:
            batch_x = np.random.uniform(size=self.shape_batches)

        return batch_x, batch_x

    def on_epoch_end(self):
        self.index = np.random.permutation(self.index)

    def __len__(self):
        return len(self.data)//self.batch_size

class WavGenerator(Sequence):
    def __init__(self,audio_metadata,batch_size=16,debug_times=False,shape_batches=None):
        self.data = audio_metadata
        self.index = np.array(self.data.index)
        self.batch_size = batch_size
        self.debug_times = debug_times
        self.shape_batches = shape_batches

    def read_wav(self,x):
        try:
            return sf.read(x['file_path'],start=x['start'],stop=x['end'])[0]
        except:
            return np.zeros((x['end']-x['start']))

    def __getitem__(self,i):
        if not self.debug_times:
            batch_idxs = np.take(self.index,np.arange(i*self.batch_size,(i+1)*self.batch_size),mode='wrap')
            batch_df = self.data.loc[batch_idxs]
            batch_x = batch_df.apply(lambda x: self.read_wav(x),axis=1)
            batch_x = np.stack(batch_x)
        else:
            batch_x = np.random.uniform(size=self.shape_batches)

        return batch_x, batch_x

    def on_epoch_end(self):
        self.index = np.random.permutation(self.index)

    def __len__(self):
        return len(self.data)//self.batch_size
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np
import pandas as pd

def get_audio_dataset(path, frame_size, hop_size):
    file_paths = list(Path(path).rglob('*.wav'))
    dfs = []
    for f in tqdm(file_paths):
        f_info = sf.info(f)
        audio_len = f_info.frames
        starts = np.arange(0,audio_len-frame_size,hop_size)
        ends = starts + frame_size
        df_i = {'start': starts,
                'end': ends,
                'file_path': [str(f.absolute())]*len(starts),
                'id': ['{}_{}'.format(f.stem,i) for i in range(len(starts))]}
        df_i = pd.DataFrame(df_i)
        dfs.append(df_i)
    out = pd.concat(dfs).set_index('id')
    return out
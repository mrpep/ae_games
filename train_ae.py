import dienen
import datasets
import generators
from IPython import embed
from callbacks import WANDBLogger
from dienen.callbacks import DecayFactorCallback
import wandb
import tensorflow as tf
import losses

ae_model = dienen.Model('models/wav_ae.yaml')
ae_model.build()
ae_model.core_model.model.summary()

nsynth_metadata = datasets.get_audio_dataset('../nsynth-valid',frame_size=4096,hop_size=4096)
#train_generator = generators.PGHIGenerator(nsynth_metadata,
#                                            batch_size=128,
#                                            frame_size=1024,
#                                            hop_size=256,
#                                            apply_log=True)

train_generator = generators.WavGenerator(nsynth_metadata,
                                          batch_size=128)
train_generator.on_epoch_end()
audio_test_data = train_generator.__getitem__(0)
#ae_model.core_model.model.compile(loss='mse',optimizer='rmsprop')
ae_model.core_model.model.compile(loss=losses.SpectralLoss(),optimizer='rmsprop')

wandb.init(name='nsynth_raw', project='ae_games',config=ae_model.core_model.model.get_config())

loggers = {'Spectrograms': {'test_data': audio_test_data,
                            'in_layers': ['wav_in'],
                            'out_layers': ['wav_in','estimated_output'],
                            'freq': 2000,
                            'unit': 'step',
                            'is_audio': True,
                            'plot_lims': [0,1]},
           'TrainMetrics': {'freq': 1, 'unit': 'step'}
          }

cbks = [WANDBLogger(loggers=loggers),
        tf.keras.callbacks.ModelCheckpoint('../ckpts_wavae/weights.{epoch:02d}-{loss:.2f}.hdf5'),
        #DecayFactorCallback('discrete_bottleneck','temperature',0.99995)
        ]
ae_model.core_model.model.fit(train_generator,callbacks=cbks,epochs=100)


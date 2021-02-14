import dienen
import datasets
import generators
from IPython import embed
from callbacks import WANDBLogger
from dienen.callbacks import DecayFactorCallback
import wandb
import tensorflow as tf

ae_model = dienen.Model('models/gsvqvae.yaml')
ae_model.build()
ae_model.core_model.model.summary()

nsynth_metadata = datasets.get_audio_dataset('../nsynth-valid',frame_size=4865,hop_size=4865)
train_generator = generators.PGHIGenerator(nsynth_metadata,
                                            batch_size=128,
                                            frame_size=1024,
                                            hop_size=256,
                                            apply_log=True)

train_generator.on_epoch_end()
audio_test_data = train_generator.__getitem__(0)
ae_model.core_model.model.compile(loss='mse',optimizer='rmsprop')

wandb.init(name='nsynth_pghi', project='ae_games',config=ae_model.core_model.model.get_config())

loggers = {'Spectrograms': {'test_data': audio_test_data,
                            'in_layers': ['spectrogram_in'],
                            'out_layers': ['scaled_estimated_spectrogram/Activation','discrete_bottleneck','translated_spectrogram'],
                            'freq': 2000,
                            'unit': 'step',
                            'is_audio': False,
                            'plot_lims': [0,1]},
           'TrainMetrics': {'freq': 1, 'unit': 'step'}
          }

cbks = [WANDBLogger(loggers=loggers),
        tf.keras.callbacks.ModelCheckpoint('../ckpts_gsvqvae/weights.{epoch:02d}-{loss:.2f}.hdf5',save_freq=2000),
        DecayFactorCallback('discrete_bottleneck','temperature',0.99995)
        ]
ae_model.core_model.model.fit(train_generator,callbacks=cbks,epochs=20)


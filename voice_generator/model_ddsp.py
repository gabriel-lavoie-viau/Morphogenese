import warnings
warnings.filterwarnings("ignore")

import os
import time
import ddsp
import ddsp.training
from ddsp.colab.colab_utils import play as audio2file
import gin
# import matplotlib.pyplot as plt
# import numpy as np
import pickle
import ddsp.spectral_ops_phonemes
import tensorflow.compat.v2 as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# # Prevent tf to grab all of the GPU memory (tf v.2.2+)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

class model_ddsp:

  def __init__(self):
    self.model = None
    self.hop_size = 64

    self.model_dir = os.path.join(os.getcwd(), 'GM-Voice')
    self.gin_file = os.path.join(self.model_dir, 'operative_config-0.gin')

  def load(self, audio_features):

    if audio_features == None:
      print("Audio features are needed to load the model")
    else:
      # LOAD AND BUILD MODEL #

      # Load the dataset statistics.
      DATASET_STATS = None
      dataset_stats_file = os.path.join(self.model_dir, 'dataset_statistics.pkl')
      print(f'Loading dataset statistics from {dataset_stats_file}')
      try:
        if tf.io.gfile.exists(dataset_stats_file):
          with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
            DATASET_STATS = pickle.load(f)
      except Exception as err:
        print('Loading dataset statistics from pickle failed: {}.'.format(err))

      # Parse gin config,
      with gin.unlock_config():
        gin.parse_config_file(self.gin_file, skip_unknown=True)

      # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
      ckpt_files = [f for f in tf.io.gfile.listdir(self.model_dir) if 'ckpt' in f]
      ckpt_name = ckpt_files[0].split('.')[0]
      ckpt = os.path.join(self.model_dir, ckpt_name)

      # Ensure dimensions and sampling rates are equal
      time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
      n_samples_train = gin.query_parameter('Additive.n_samples')
      self.hop_size = int(n_samples_train / time_steps_train)

      time_steps = int(audio_features['num_samples'] / self.hop_size) # #of sample in audio
      n_samples = time_steps * self.hop_size

      print("===Trained model===")
      print("Time Steps", time_steps_train)
      print("Samples", n_samples_train)
      print("Hop Size", self.hop_size)
      print("\n===Resynthesis===")
      print("Time Steps", time_steps)
      print("Samples", n_samples)
      print('')

      gin_params = [
          'RnnFcDecoder.input_keys = ("ld_scaled","f0_scaled","z")',
          'Additive.n_samples = {}'.format(n_samples),
          'FilteredNoise.n_samples = {}'.format(n_samples),
          'DefaultPreprocessor.time_steps = {}'.format(time_steps),
      ]

      # print(gin.query_parameter('ProcessorGroup.dag'))

      with gin.unlock_config():
        gin.parse_config(gin_params)

      # print(audio_features)

      # Trim all input vectors to correct lengths 
      for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]

      # Set up the model just to predict audio given new conditioning
      self.model = ddsp.training.models.Autoencoder()
      self.model.restore(ckpt)

      #---------------------------------------------------------------

      # Build model by running a batch through it.
      start_time = time.time()
      _ = self.model(audio_features, training=False)
      print('Restoring model took %.1f seconds' % (time.time() - start_time))
      print('\n')

  def predict(self, audio_features, voice_name):

    # Run a batch of predictions.
    start_time = time.time()
    
    # Update preprocessor and processors params
    time_steps = int(audio_features['num_samples'] / self.hop_size)
    n_samples = time_steps * self.hop_size

    self.model.preprocessor.time_steps = time_steps
    self.model.processor_group.processors[0].n_samples = n_samples
    self.model.processor_group.processors[1].n_samples = n_samples

    # print(self.model.processor_group.processors[0])
    # for keys in self.model.processor_group.processors.__dict__.keys():
    #   print('\n', keys)

    # Trim all input vectors to correct lengths 
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
      audio_features[key] = audio_features[key][:time_steps]

    # Generate audio
    audio_gen = self.model(audio_features, training=False)

    print('\nAudio synthesized!')
    print('Prediction took %.1f seconds\n' % (time.time() - start_time))

    # Export audio file to disk
    filename = voice_name + '.wav'
    filepath = os.path.join('./generated', filename)
    audio2file(audio_gen, filename=filepath)

    # Path relative to pure data main
    filepath = os.path.join('../voice_generator/generated', filename)
    return filepath
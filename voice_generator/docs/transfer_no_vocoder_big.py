# Ignore a bunch of deprecation warnings
import warnings
warnings.filterwarnings("ignore")

import copy
import os
import time

import crepe
import ddsp
import ddsp.training
from ddsp.colab import colab_utils
from ddsp.colab.colab_utils import (
    auto_tune, detect_notes, fit_quantile_transform, 
    get_tuning_factor, 
    #download, 
    play, record, 
    specplot, upload, DEFAULT_SAMPLE_RATE)
import gin
# from google.colab import files
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import ddsp.spectral_ops_phonemes

from OSC_server import *

# Helper Functions
sample_rate = 16000

print('Imports Done!')

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

# osc = OSC_server()
# osc.start()

# for key in os.environ:
#   print(key)

# UPLOAD AUDIO AND EXTRACT FEATURES #

# Upload Audio
# filename = 'gaelic_softer.wav'
# filename = 'gaelic_short_16khz.wav'
filename = 'm1_short_16khz.wav'
filepath = '/media/gabriel/Extra/ddsp_gm2/experiments/audio_samples/' + filename
audio = upload(filepath)
audio = audio[np.newaxis, :]
print('\nExtracting audio features...')

# Plot.
# specplot(audio)
play(audio, filename='./exports/' + 'original_' + filename)

# Setup the session.
ddsp.spectral_ops.reset_crepe()

# Compute features.
start_time = time.time()
audio_features = ddsp.training.metrics.compute_audio_features_with_phonemes(audio)
audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
audio_features['phoneme'] = audio_features['phoneme'].astype(np.float32)
audio_features_mod = None
print('Audio features took %.1f seconds' % (time.time() - start_time))
#deletme still numpy

print('\n')
print(audio_features.keys())
print('\n')
print('Audio        : ', audio_features['audio'].shape)  # All of the audio smaples (16Khz)
print('Loudness (db): ', audio_features['loudness_db'].shape)
print('F0 (hz)      : ', audio_features['f0_hz'].shape)
print('F0 confidence: ', audio_features['f0_confidence'].shape)
print('Phoneme      : ', audio_features['phoneme'].shape)
print('\n')

TRIM = -15
# # Plot Features
# fig, ax = plt.subplots(nrows=3, 
#                        ncols=1, 
#                        sharex=True,
#                        figsize=(6, 8))
# ax[0].plot(audio_features['loudness_db'][:TRIM])
# ax[0].set_ylabel('loudness_db')

# ax[1].plot(librosa.hz_to_midi(audio_features['f0_hz'][:TRIM]))
# ax[1].set_ylabel('f0 [midi]')

# ax[2].plot(audio_features['f0_confidence'][:TRIM])
# ax[2].set_ylabel('f0 confidence')
# _ = ax[2].set_xlabel('Time step [frame]')

# # plt.show()

temp_phonemes = list(audio_features['phoneme'])
# https://cmusphinx.github.io/
with open('./audio_features/loudness_python.txt', 'w') as f:
    for item in audio_features['loudness_db']:
        f.write("%s " % item)

with open('./audio_features/f0_python.txt', 'w') as f:
    for item in audio_features['f0_hz']:
        f.write("%s " % item)

with open('./audio_features/f0_confidence_python.txt', 'w') as f:
    for item in audio_features['f0_confidence']:
        f.write("%s " % item)

with open('./audio_features/phoneme_python.txt', 'w') as f:
    for item in temp_phonemes:
        f.write("%s " % item)


# #----------------------------------------------------------------------
# #----------------------------------------------------------------------

# LOAD AND BUILD MODEL #

# model_dir = find_model_dir('./GM-Voice/')
model_dir = os.path.join(os.getcwd(), 'GM-Voice')
gin_file = os.path.join(model_dir, 'operative_config-0.gin')

# print("GIN FILE : ", gin_file)

# Load the dataset statistics.
DATASET_STATS = None
dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
print(f'Loading dataset statistics from {dataset_stats_file}')
try:
  if tf.io.gfile.exists(dataset_stats_file):
    with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
      DATASET_STATS = pickle.load(f)
except Exception as err:
  print('Loading dataset statistics from pickle failed: {}.'.format(err))


# Parse gin config,
with gin.unlock_config():
  gin.parse_config_file(gin_file, skip_unknown=True)

# Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
ckpt_name = ckpt_files[0].split('.')[0]
ckpt = os.path.join(model_dir, ckpt_name)

print("CKPT : ", ckpt)
print('\n')

# Ensure dimensions and sampling rates are equal
time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
# time_steps_train = gin.query_parameter('PhonemePreprocessor.time_steps')
n_samples_train = gin.query_parameter('Additive.n_samples')
hop_size = int(n_samples_train / time_steps_train)

time_steps = int(audio.shape[1] / hop_size)
n_samples = time_steps * hop_size

print("===Trained model===")
print("Time Steps", time_steps_train)
print("Samples", n_samples_train)
print("Hop Size", hop_size)
print("\n===Resynthesis===")
print("Time Steps", time_steps)
print("Samples", n_samples)
print('')

gin_params = [
    # 'RnnFcDecoder.input_keys = ("ld_scaled","f0_scaled","z","phoneme")',
    'RnnFcDecoder.input_keys = ("ld_scaled","f0_scaled","z")',
    'Additive.n_samples = {}'.format(n_samples),
    'FilteredNoise.n_samples = {}'.format(n_samples),
    # 'PhonemePreprocessor.time_steps = {}'.format(time_steps),
    'DefaultPreprocessor.time_steps = {}'.format(time_steps),
]

with gin.unlock_config():
  gin.parse_config(gin_params)

# Trim all input vectors to correct lengths 
# for key in ['f0_hz', 'f0_confidence', 'loudness_db','phoneme']:
for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
  audio_features[key] = audio_features[key][:time_steps]
#   audio_features[key] = audio_features[key][:, np.newaxis]
audio_features['audio'] = audio_features['audio'][:, :n_samples]

# for k, v in audio_features.items():
#   print(k, ': ', v.shape)
# print('\n')

# Set up the model just to predict audio given new conditioning
model = ddsp.training.models.Autoencoder()
model.restore(ckpt)

# Build model by running a batch through it.
start_time = time.time()
_ = model(audio_features, training=False)
print('Restoring model took %.1f seconds' % (time.time() - start_time))
print('\n')

#----------------------------------------------------------------------
#----------------------------------------------------------------------

# ADJUST THE AUDIO FEATURES #

# These models were not explicitly trained to perform timbre transfer, 
# so they may sound unnatural if the incoming loudness and frequencies 
# are very different then the training data (which will always be somewhat true). 


# Note Detection

# You can leave this at 1.0 for most cases
threshold = 1.0 # min: 0.0, max:2.0

# Automatic
ADJUST = False # type:"boolean"

# Quiet parts without notes detected (dB)
quiet = 40 # min: 0, max:60

# Force pitch to nearest note (amount)
autotune = 0.0 # min: 0.0, max:1.0

# Manual
# Shift the pitch (octaves)
pitch_shift =  0 # min:-2, max:2

# Adjsut the overall loudness (dB)
loudness_shift = 0 # min:-20, max:20

# for k, v in audio_features.items():
#   print(k)
#   print(v.shape)
audio_features_mod = {k: v.copy() for k, v in audio_features.items()}


## Helper functions.
def shift_ld(audio_features, ld_shift=0.0):
  """Shift loudness by a number of octaves."""
  audio_features['loudness_db'] += ld_shift
  return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
  """Shift f0 by a number of ocatves."""
  audio_features['f0_hz'] *= 2.0 ** (pitch_shift)
  audio_features['f0_hz'] = np.clip(audio_features['f0_hz'], 
                                    0.0, 
                                    librosa.midi_to_hz(110.0))
  return audio_features


mask_on = None

if ADJUST and DATASET_STATS is not None:
  # Detect sections that are "on".
  mask_on, note_on_value = detect_notes(audio_features['loudness_db'],
                                        audio_features['f0_confidence'],
                                        threshold)
  
  if np.any(mask_on):
    # Shift the pitch register.
    target_mean_pitch = DATASET_STATS['mean_pitch']
    pitch = ddsp.core.hz_to_midi(audio_features['f0_hz'])
    mean_pitch = np.mean(pitch[mask_on])
    p_diff = target_mean_pitch - mean_pitch
    p_diff_octave = p_diff / 12.0
    round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
    p_diff_octave = round_fn(p_diff_octave)
    audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)


    # Quantile shift the note_on parts.
    _, loudness_norm = colab_utils.fit_quantile_transform(
        audio_features['loudness_db'],
        mask_on,
        inv_quantile=DATASET_STATS['quantile_transform'])

    # Turn down the note_off parts.
    mask_off = np.logical_not(mask_on)
    loudness_norm[mask_off] -=  quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
    loudness_norm = np.reshape(loudness_norm, audio_features['loudness_db'].shape)
    
    audio_features_mod['loudness_db'] = loudness_norm 

    # Auto-tune.
    if autotune:
      f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod['f0_hz']))
      tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
      f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=autotune)
      audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)

  else:
    print('\nSkipping auto-adjust (no notes detected or ADJUST box empty).')

else:
  print('\nSkipping auto-adujst (box not checked or no dataset statistics found).')

# Manual Shifts.
audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
audio_features_mod = shift_f0(audio_features_mod, pitch_shift)


# Plot Features.
has_mask = int(mask_on is not None)
n_plots = 3 if has_mask else 2 
fig, axes = plt.subplots(nrows=n_plots, 
                      ncols=1, 
                      sharex=True,
                      figsize=(2*n_plots, 8))

if has_mask:
  ax = axes[0]
  ax.plot(np.ones_like(mask_on[:TRIM]) * threshold, 'k:')
  ax.plot(note_on_value[:TRIM])
  ax.plot(mask_on[:TRIM])
  ax.set_ylabel('Note-on Mask')
  ax.set_xlabel('Time step [frame]')
  ax.legend(['Threshold', 'Likelihood','Mask'])

ax = axes[0 + has_mask]
ax.plot(audio_features['loudness_db'][:TRIM])
ax.plot(audio_features_mod['loudness_db'][:TRIM])
ax.set_ylabel('loudness_db')
ax.legend(['Original','Adjusted'])

ax = axes[1 + has_mask]
ax.plot(librosa.hz_to_midi(audio_features['f0_hz'][:TRIM]))
ax.plot(librosa.hz_to_midi(audio_features_mod['f0_hz'][:TRIM]))
ax.set_ylabel('f0 [midi]')
_ = ax.legend(['Original','Adjusted'])

# plt.show()

# #----------------------------------------------------------------------
# #----------------------------------------------------------------------

# RESYNTHESIZE THE AUDIO

af = audio_features if audio_features_mod is None else audio_features_mod

for k, v in af.items():
  print(k, ': ', v.shape, type(v))
print('\n')

# Run a batch of predictions.
start_time = time.time()
audio_gen = model(af, training=False)
print('Prediction took %.1f seconds' % (time.time() - start_time))

# Plot
play(audio_gen, filename='./exports/' + 'python_' + filename)

print('Audio synthesized!')

# specplot(audio)
# plt.title("Original")

# specplot(audio_gen)
# _ = plt.title("Resynthesis")

# plt.show()

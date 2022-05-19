import os
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Model,load_model
import pyaudio
import wave

import time

chunk = 1024

sample_format = pyaudio.paInt16
chanels = 1
smpl_rt = 44100
seconds = 4
pa = pyaudio.PyAudio()

stream = pa.open(format=sample_format, channels=chanels,
                 rate=smpl_rt, input=True,
                 frames_per_buffer=chunk)

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_waveform_and_label(file_path):
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform

AUTOTUNE = tf.data.AUTOTUNE

def get_spectrogram(waveform):
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  waveform = tf.cast(waveform, dtype=tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def get_spectrogram_and_label_id(audio):
  spectrogram = get_spectrogram(audio)
  return spectrogram

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
  return output_ds

# Carrega Modelo RNA
model = load_model('rna_TESTE13.h5')

def record():
  stream.start_stream()
  print(pa.get_default_input_device_info())
  print('Recording...')
  frames = []
  for i in range(0, int(smpl_rt / chunk * seconds)):
       data = stream.read(chunk)
       frames.append(data)
  stream.stop_stream()
  

  print('Done !!! ')
  sf = wave.open("filename.wav", 'wb')
  sf.setnchannels(chanels)
  sf.setsampwidth(pa.get_sample_size(sample_format))
  sf.setframerate(smpl_rt)
  sf.writeframes(b''.join(frames))
  sf.close()
  return 'filename.wav'

#Passa a função de gravação como parametro
#sample_file = record()


#Processamento do  audio
#sample_ds = preprocess_dataset([str(sample_file)])
#prediction = model.predict(sample_ds.batch(1))

#Previsão
def previ(variavel):
    if (prev[0] > 0.50) and (prev[0] > a[1]) and (prev[0] > a[2]):
       print("BUZINA")
    if (prev[1] > 0.50) and (prev[1] > a[0]) and (prev[1] > a[2]):
        print("SIRENE")
    if (prev[2] > 0.50) and (prev[2] > a[0]) and prev[2] > a[1]):
        print("APITO")
  return

x = 0
while (x <= 1):
    sample_file = record()
    sample_ds = preprocess_dataset([str(sample_file)])
    prediction = model.predict(sample_ds.batch(1))
    prev = tf.nn.softmax(prediction[0])
    previ(prev)


    x = 1

stream.close()
pa.terminate()
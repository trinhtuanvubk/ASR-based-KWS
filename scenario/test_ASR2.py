import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig
import copy
import pytorch_lightning as pl
import IPython
import torch

# File wav
# Audio_sample = 'data/test-clean/6930/76324/6930-76324-0022.wav'
Audio_sample = '/home/trinhtuanvu/Phoneme/wav/19-198-0001.wav'
IPython.display.Audio(Audio_sample)

# Load model
# model_path = "./lightning_logs/version_8/checkpoints/epoch=4-step=17839.ckpt"
model_path = "../lightning_logs/version_7/checkpoints/epoch=2-step=21404.ckpt"

asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=model_path)

# Convert our audio sample to text
files = [Audio_sample]
raw_text = ''
text = ''
for fname, transcription in zip(files, asr_model.transcribe(paths2audio_files=files, logprobs=True)):
  raw_text = transcription

# vocab = asr_model.decoder.vocabulary + ['-']
vocab = asr_model.decoder.vocabulary
print(vocab)

print(raw_text.shape)
print(raw_text.sum(axis=1).shape)
# for i in raw_text.sum(axis=1):
# 	print(i)

greedy = raw_text.argmax(dim=-1, keepdim=False)
print(greedy)
print(greedy.shape)

print([vocab[int(_)] for _ in greedy])

print(len(vocab))
for fname, transcription in zip(files, asr_model.transcribe(paths2audio_files=files, logprobs=0)):
  raw_text = transcription
print(raw_text, len(raw_text))
a = raw_text.split(" ")
a =" ".join(a)
print(a)
# import pandas as pd 
# import numpy as np
# pd.DataFrame(np.exp(raw_text.numpy())).to_csv("test.csv")
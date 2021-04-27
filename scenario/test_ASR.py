import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig
import copy
import pytorch_lightning as pl
import IPython
import math
import torch.tensor 
import inspect

# File wav
# Audio_sample = '/home/trinhtuanvu/Phoneme/wav/19-198-0001.wav'
Audio_sample = '/home/trinhtuanvu/Phoneme/wav/test.wav'

# IPython.display.Audio(Audio_sample)

# Load model
# model_path = "./lightning_logs/version_8/checkpoints/epoch=4-step=17839.ckpt"
model_path = "../lightning_logs/version_7/checkpoints/epoch=2-step=21404.ckpt"

asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=model_path)

# Convert our audio sample to text
files = [Audio_sample]
raw_text = ''
text = ''
for fname, transcription in zip(files, asr_model.transcribe(paths2audio_files=files,logprobs = 1)):
  raw_text = transcription
#  asr_model.transcribe(paths2audio_files=files,logprobs = 0)):
phoneme = asr_model.transcribe(paths2audio_files=files,logprobs = 0)
print(inspect.getsource(asr_model.transcribe))
print(len(asr_model.decoder.vocabulary))
print(torch.exp(raw_text[-1,:]))
print(raw_text)
print(phoneme,"len", len(phoneme))
print(raw_text.shape)
print(raw_text.argmax(dim = -1))
# print(len([i] for i in (raw_text.argmax(dim = -1)).tolist() if i != 0))
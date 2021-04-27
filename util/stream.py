from scipy.io import wavfile
from scipy import signal
from queue import Queue
import numpy as np
import pyaudio
import random
import torch
import time
import math
import sys
import io
import os
import wave
# from playsound import playsound
import util

class stream_audio():
    def __init__(self, model, #args,
                 sample_rate=16000,
                 chunk_duration=0.25,
                 feed_duration=1.0,
                 channels=1,
                 pause=False):

        # self.args = args
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.feed_duration = feed_duration
        self.channels = channels
        
        # From device 
        self.device_sample_rate = 44100
        self.chunk_samples = int(self.device_sample_rate * self.chunk_duration)
        self.feed_samples = int(self.device_sample_rate * self.feed_duration)
        
        # Queue to communiate between the audio callback and main thread
        self.q = Queue()
        
        # Data buffer for the input wavform
        self.data = np.zeros(self.feed_samples, dtype='int16')
        
        # Setting
        self.stream = True
        self.pause = False
        self.model = model
    
    def run(self):
        '''
        FIRST: Define your keyword (anchor)
        '''
        print('FIRST: Say your keyword {} times'.format("3"))#format(self.args.num_anchor))
        self.path = []
        for i in range(3) :#(self.args.num_anchor):
            input('Press enter to record : ')
            util.record()
            # _, signal = wavfile.read('record/example.wav')
            # mel, _ = util.fbank(signal, samplerate=16000, 
            #                     winlen=0.025, winstep=0.012, nfilt=64)
            # self.anchor.append(mel)
            ex_path = "record/example.wav"
            self.path.append(ex_path)
        # self.anchor = torch.Tensor(np.array(self.anchor))
        # self.anchor = self.model(self.anchor)
        self.matrix_learned_phoneme,self.matrix_w,self.matrix_probs = util.create_database_ctc(self.path)
        
        self.lower_boundary, self.upper_boundary = util.threshold_ctc(self.matrix_learned_phoneme,self.matrix_w,self.matrix_probs)
        print('Confidence interval:', self.lower_boundary, self.upper_boundary)
        '''
        NEXT: Streaming
        '''
        # Callback method
        def audio_callback(in_data, frame_count, time_info, status):
            data0 = np.frombuffer(in_data, dtype='int16')
            self.data = np.append(self.data, data0)    
            if len(self.data) > self.feed_samples:
                self.data = self.data[-self.feed_samples:]
                # Process data async by sending a queue.
                self.q.put(self.data)
            return (in_data, pyaudio.paContinue)
             
        # Open port audio
        self.audio = pyaudio.PyAudio()
        self.stream_in = self.audio.open(input=True, output=False,
                                         format=pyaudio.paInt16,
                                         channels=self.channels,
                                         rate=self.device_sample_rate,
                                         frames_per_buffer=self.chunk_samples,
                                         stream_callback=audio_callback)
        
        try:
            while self.stream:             
                data = self.q.get()
                # path = self.save_wavefile(data)
                file_name = "record/now.wav"
               
                
                path_wavefile = util.save_wavefile(data = data,
                                                   file_name = file_name,
                                                   stream = self.audio,
                                                   channels = self.channels,
                                                   sample_format = pyaudio.paInt16,
                                                   sample_rate = self.sample_rate)
                threshold = util.score_ctc(path = path_wavefile,
                                           matrix_learned_phoneme = self.matrix_learned_phoneme,
                                           matrix_w =  self.matrix_w)
                print(threshold)
                if threshold > self.upper_boundary:
                    # playsound('record/tingting.wav')
                    print('Predict: Keyword')
                else:
                    print('Predict: Non-keyword')

        except (KeyboardInterrupt, SystemExit):
            self.stream = False
            self.stream_in.stop_stream()
            self.stream_in.close()
            self.audio.terminate()
    # # def save_wavefile()
    # def save_wavfile(self,data) :
    #     file_name = "record/now.wav"
    #     self.wf = wave.open(file_name,'wb')
    #     self.wf.setnchannels(self.channels)
    #     self.wf.setsampwidth(self.audio.get_sample_size(sample_format))
    #     self.wf.setframerate(self.sample_rate)
    #     self.wf.writeframes(b''.join(data))
    #     self.wf.close()
    #     return file_name
    # def predict(self, data):
    #     # Downsampling with melform
    #     data = signal.resample(data, self.sample_rate)
    #     example, _ = util.fbank(data, samplerate=16000, 
    #                      winlen=0.025, winstep=0.012, nfilt=64)
    #     example = np.expand_dims(example, axis=0)
        
    #     # Compute threshold
    #     example = self.model(torch.Tensor(example)).reshape(-1)
    #     anchor = self.anchor[random.randint(0, self.args.num_anchor-1)]
    #     threshold = torch.norm(example - anchor, p=2)
    #     return threshold
        
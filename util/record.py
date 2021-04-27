import pyaudio
import wave 

def record() : 
    chunk = 1000 # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16 # 16 bits per sample
    channels = 1 
    fs = 16000 # Record at 44100 samples per second 
    seconds = 2
    filename = "record/example.wav" 

    p = pyaudio.PyAudio() # Create an interface to PortAudio

    print('RECORDING.....')
    # p.going() = True

    stream = p.open(format = sample_format,
                    channels =channels,
                    rate = fs,
                    frames_per_buffer = chunk,
                    input = True)
                    # input_device_index = 1)
    frames = [] #initialize array to store frames 

    # Store data in chunks for 3 seconds 
    for i in range(0, int(fs / chunk * seconds)) : 
        data = stream.read(chunk)
        frames.append(data)
    
    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    #terminate the PortAudio interface 
    p.terminate()

    print('FINISHED')

    # Save record dato as a wave file 
    wf = wave.open(filename,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


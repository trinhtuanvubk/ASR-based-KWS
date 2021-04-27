import wave 
'''
This function is to save wave file to transcribe 
'''
def save_wavefile(data,file_name,stream,channels,sample_format,sample_rate) : 
    wf = wave.open(file_name,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(stream.get_sample_size(sample_format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(data))
    wf.close()
    return file_name
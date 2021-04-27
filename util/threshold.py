import util
import numpy as np
import math

              
def threshold_ctc(matrix_learned_phoneme, matrix_w, matrix_probs) : 
    scores = []
    # all_logprobs = []
    print(matrix_learned_phoneme)
    # print(matrix_probs)
    for phoneme in matrix_learned_phoneme : 
        logprobs = []
        for pr in matrix_probs :
            probs = util.CTCforward(learned_phoneme = phoneme, matrix = pr)
            logprobs.append(np.log(probs)) #need to log 

        # all_logprobs.append()
        score = sum([logprobs[i]*matrix_w[i] for i in range(len(matrix_w))])
        # print([matrix_w[i].tolist() for i in range(len(matrix_w))])
        scores.append(score)
        # print(score)
    # print(matrix_w)
    mean = np.mean(scores)
    
    std = np.std(scores)
    interval =  interval = 1.96 * std / math.sqrt(len(scores))
    print(scores)
    print("mean",mean)
    print("std",std)
    print("interval",interval)

    upper_boundary = mean + interval
    lower_boundary = mean - 2*interval
    print("upper_ : ", upper_boundary)
    print("lower_ : ", lower_boundary)
    return upper_boundary,lower_boundary
# from scipy.io import wavfile 
# sample_rate , signal = wavfile.read("/home/trinhtuanvu/Phoneme/wav/0b09edd3_nohash_0.wav")



        

        
        
    



from tqdm import tqdm 
import numpy as np 
import math 
import time 
import nemo.collections.asr as nemo_asr

import util 


def inference_ctc(matrix_learned_phoneme,matrix_w,matrix_probs) : 
    # Load model from checkpoint 
    model_path = "./lightning_logs/version_7/checkpoints/epoch=2-step=21404.ckpt"
    asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path = model_path)#args = arg
    
    # load saved parameters 
    # matrix_greedy, matrix_w = util.load_database() 
    
    # chose threshold
    upper_boundary,lower_boundary = util.threshold_ctc(matrix_learned_phoneme,matrix_w, matrix_probs)
    print(upper_boundary,lower_boundary)
    threshold = lower_boundary
    while(1) : 
        input('press enter to record :')
        util.record()
        path = "record/example.wav"
        files = [path]
        score = util.score_ctc(path,matrix_learned_phoneme,matrix_w)
        
        for fname, trans in zip(files, asr_model.transcribe(paths2audio_files=files, logprobs=0)) : 
            phoneme = trans
        print(phoneme,"\n")
        print("threshold",threshold)
        print("score",score)
        if score > threshold : 
            print("KEYWORD")
        else :
            print("NON-KEYWORD")
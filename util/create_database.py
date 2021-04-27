import nemo 
import nemo.collections.asr as nemo_asr 
import torch 
import numpy as np 
import torch.tensor
import math
import util
import numpy as np 


# def create_dict_c(*args) : 
#     return((k, eval(k)) for k in args)

def create_database_ctc(args) : 
    # load model 
    model_path    = "./lightning_logs/version_7/checkpoints/epoch=2-step=21404.ckpt"
    asr_model     = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path=model_path)
    matrix_w      = []
    matrix_learned_phoneme = []
    matrix_probs = []
    for file in args : 
        files = [file]

        # tensor of the probs(log) each step
        for fname, probs in zip(files, asr_model.transcribe(paths2audio_files=files, logprobs=1)):
            tensor_probs  = probs
        for fname, phoneme in zip(files, asr_model.transcribe(paths2audio_files=files, logprobs=0)):
            learned_phoneme = phoneme
        
        matrix_learned_phoneme.append(learned_phoneme)
        # print(tensor_probs)
        # the probs of learned_phoneme 
        # learned_phoneme_probs = math.prod(torch.max(np.exp(tensor_probs),1).values)
        learned_phoneme_probs = util.CTCforward(learned_phoneme,np.exp(tensor_probs.numpy()))
        matrix_probs.append(np.exp(tensor_probs.numpy()))
        # print(matrix_probs,len(matrix_probs),len(matrix_probs[0]))
        # print(np.max(matrix_probs,axis = 0 ))

        # weight (confidence of learned_phoneme)	
        w = -1 / np.log(learned_phoneme_probs)
        matrix_w.append(w)
    # print("matrix_w : ",matrix_w)
    return matrix_learned_phoneme,matrix_w,matrix_probs
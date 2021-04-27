# import nemo 
import nemo.collections.asr as nemo_asr 
import torch 
import numpy as np 
import torch.tensor
# import math
import util

def score_stream(data,matrix_learned_phoneme,matrix_w):
    model_path = "./lightning_logs/version_7/checkpoints/epoch=2-step=21404.ckpt"
    asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path = model_path)
    # calculate score 
    # files = [path]
    logprobs = []
    # for fname, prob in zip(files, asr_model.transcribe(paths2audio_files = files , logprobs=1)) :
        # tensor_probs = (torch.exp(probs).numpy()).tolist()
        # tensor_probs = prob
    # print(tensor_probs)
    tensor_probs = asr_model.transcribe(data,logprobs=1)
    tensor_probs = np.exp(tensor_probs.numpy())
    # print(tensor_probs)
    # print(np.max(tensor_probs,axis=1))
        
        # print(type(tensor_probs[0]))
        # tensor_probs = [np.exp(i) for i in tensor_probs]
    

    for phoneme in matrix_learned_phoneme:
        probs = util.CTCforward(learned_phoneme = phoneme,matrix = tensor_probs)
        # print(probs)
        logprobs.append(np.log(probs))
        # print(type(logprobs[0]))
    # score 
    # score = sum([logprobs[i]*matrix_w[i]] for i in range(len(matrix_w)))
    score = sum([i*j for i,j in zip(logprobs,matrix_w)])
    return score 
 



def score_ctc(path, matrix_learned_phoneme, matrix_w) :
    model_path = "./lightning_logs/version_7/checkpoints/epoch=2-step=21404.ckpt"
    asr_model = nemo_asr.models.EncDecCTCModel.load_from_checkpoint(checkpoint_path = model_path)
    # calculate score 
    files = [path]
    logprobs = []
    for fname, prob in zip(files, asr_model.transcribe(paths2audio_files = files , logprobs=1)) :
        # tensor_probs = (torch.exp(probs).numpy()).tolist()
        tensor_probs = prob
    # print(tensor_probs)
    tensor_probs = np.exp(tensor_probs.numpy())
    # print(tensor_probs)
    # print(np.max(tensor_probs,axis=1))
        
        # print(type(tensor_probs[0]))
        # tensor_probs = [np.exp(i) for i in tensor_probs]
    

    for phoneme in matrix_learned_phoneme:
        probs = util.CTCforward(learned_phoneme = phoneme,matrix = tensor_probs)
        # print(probs)
        logprobs.append(np.log(probs))
        # print(type(logprobs[0]))
    # score 
    # score = sum([logprobs[i]*matrix_w[i]] for i in range(len(matrix_w)))
    score = sum([i*j for i,j in zip(logprobs,matrix_w)])
    return score 
 

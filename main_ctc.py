import scenario 
import util 

path1 = '../Phoneme/wav/lumi.wav'
path2 = '../Phoneme/wav/lumi-1.wav'
path3 = '../Phoneme/wav/lumi-2.wav'
path = []

path.append(path1)
path.append(path2)
path.append(path3)
print(path)

matrix_learned_phoneme,matrix_w,matrix_probs = util.create_database_ctc(path)

scenario.inference_ctc(matrix_learned_phoneme,matrix_w,matrix_probs)

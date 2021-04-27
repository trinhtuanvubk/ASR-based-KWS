import scenario 
import util 

# path1 = '/home/trinhtuanvu/Phoneme/wav/0b09edd3_nohash_0.wav' 

# path1 = '/home/trinhtuanvu/Phoneme/wav/0ab3b47d_nohash_0.wav'
path1 = '../Phoneme/wav/lumi.wav'
path2 = '../Phoneme/wav/lumi-1.wav'
path3 = '../Phoneme/wav/lumi-2.wav'
path = []

path.append(path1)
path.append(path2)
path.append(path3)
print(path)
# path4 = '/home/trinhtuanvu/Phoneme/wav/0ab3b47d_nohash_0.wav' 
# path3 = '/home/trinhtuanvu/Phoneme/wav/0ab3b47d_nohash_0.wav' 
matrix_learned_phoneme,matrix_w,matrix_probs = util.create_database_ctc(path)
# print("matrix_learned_phoneme :", matrix_learned_phoneme, len(matrix_learned_phoneme),type(matrix_learned_phoneme))
# print("matrix_probs :",matrix_probs, len(matrix_probs),type(matrix_probs))
# print("matrix_w ",matrix_w,len(matrix_w),type(matrix_w))
# matrix_greedy, matrix_w = util.load_database()
# print(matrix_probs)


# print(matrix_beam.shape,matrix_w.shape,matrix_probs.shape)

scenario.inference_ctc(matrix_learned_phoneme,matrix_w,matrix_probs)

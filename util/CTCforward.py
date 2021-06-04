
import numpy as np 
# create CTC forward fucntion 
def CTCforward(learned_phoneme, matrix):
    # matrix of inference file (probs version ,not log version )
    vocabs = [' ', 'UW1', 'AW1', 'F', 'UW2', 'AO0', 'EY1', 'V',\
     'ER0', 'NG', 'AY1', 'P', 'UH1', 'EH0', 'ER2', 'sil', 'DH',\
     'S', 'EH2', 'IH2', 'AA1', 'CH', 'sp', 'AE0', 'EY0', 'TH',\
     'ER1', 'HH', 'OW1', 'EH1', 'OY1', 'AO2', 'AA0', 'K', \
     'AA2', 'AY2', 'UH0', 'IH1', 'AW2', 'B', 'T', 'W', 'M', 'IY2', 'R', \
     'SH', 'OY0', 'IY0', 'Y', 'AH1', 'AE1', 'D', 'AH0', 'UW0', 'OW2', 'N', \
     'UH2', 'AO1', 'Z', 'JH', 'IH0', 'G', 'AE2', 'AW0', 'IY1', 'OW0', 'OY2', \
     'AH2', 'AY0', 'ZH', 'spn', 'EY2', 'L','-']

    phoneme_list = learned_phoneme.split(" ")
    l = []
    for i in phoneme_list :
        l.append("-")
        l.append(i)
        l.append("-")
        l.append(" ")
    l.pop()

    phoneme_list = list(set(phoneme_list))
    phoneme_list.extend([" ","-"])
    idx_col = []
    for ele in l :
        idx_col.append(vocabs.index(ele))
    # print(idx_col)
    matrix = np.array(matrix).T
    # print("shape",np.shape(matrix))
    # print(l)
    
    # print(matrix_labels)
    matrix_l = np.array([matrix[i] for i in idx_col])

    for i in range(1,matrix_l.shape[1]) :
        for j in range(matrix_l.shape[0]) : 
            p = matrix_l[j,i]
            if l[j]=='-' or(j>1 and l[j]==l[j-2])  :
                matrix_l[j,i] = (matrix_l[j,i-1] + matrix_l[j-1,i-1]) * p
            else : 
                matrix_l[j,i] = (matrix_l[j,i-1] + matrix_l[j-1,i-1] + matrix_l[j-2,i-1]) * p
            # print(matrix_l[j,i])
    # print(matrix_l[-2,-1] + matrix_l[-1,-1])
    return matrix_l[-2,-1] + matrix_l[-1,-1]
    

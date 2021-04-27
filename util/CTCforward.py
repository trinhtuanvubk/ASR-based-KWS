
import numpy as np 
# def alpha(t , s , matrix_l ,phoneme_list) :
#     # t,s < 0 return 0 
#     if t < 0 or s < 0:
#         return 0
#         #at timestep 0 , must begin with "-" or first char 
#     if t == 0:
#         if s == 0:
#             return matrix_l[phoneme_list.index(' ')][0]
#         if s == 1:
#             return matrix_l[phoneme_list.index(l[1])][0]
#         return 0
    
#     if 2 * s + 1 < t or 2 * t + 1 < s:
#         return 0
    
#     def _alpha():
#         return alpha(t-1, s) + alpha(t-1, s-1)
    
#     p = matrix_l[phoneme_list.index(l[s])][t]
#     if l[s] == '-' or (s > 1 and l[s] == l[s-2]):
#         return _alpha() * p    
#     return (_ alpha() + alpha(t-1, s-2)) * p
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
    # matrix_l = np.array()
    # for i in idx_col : 
    matrix_l = np.array([matrix[i] for i in idx_col])

    # matrix_l[2:,0] = 0
    # matrix_l[:,-1] = 0
    # print(np.max(matrix_l,axis=1))
    # import pandas
    # df = pandas.DataFrame(matrix_l)
    # df.to_csv("data.csv")
    # print(matrix_l)
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
    

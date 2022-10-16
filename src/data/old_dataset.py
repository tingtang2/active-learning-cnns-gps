from typing import Tuple
import numpy as np

nuc_arr = ['A','C','G','T']

#Function for calculating modified probability of splicing at SD1
def prob_SD1 (sd1_freq: float, sd2_freq: float) -> float:
    if (sd1_freq == 0 and sd2_freq == 0):
        return 0.0
    else:
        return sd1_freq/(sd1_freq + sd2_freq)

#Function converting nucleotide sequence to numerical array with 4 channels
def seq_to_arr(seq: str) -> np.ndarray:
    seq_len = len(seq)
    arr_rep = np.zeros((seq_len, len(nuc_arr)))
    for i in range(seq_len):
        arr_rep[i][nuc_arr.index(seq[i])] = 1
    return arr_rep

# Storing model inputs (DNA sequences) and outputs (probability of splicing at SD1)
def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    dataset_path = "../5SS_compressed.txt"
    seq_len = 101
    n = 265137
    inputs = np.zeros((n,seq_len, 4))
    prob_s1 = np.zeros(n)

    with open(dataset_path) as f:
        ind = 0
        for line in f:
            mod_line = line.split('\t')
            inputs[ind] = seq_to_arr(mod_line[1])
            prob_s1[ind] = prob_SD1(float(mod_line[2]), float(mod_line[3][:-1]))
            ind += 1
    
    return np.array(inputs), np.array(prob_s1)
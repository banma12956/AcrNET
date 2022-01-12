import os
import copy
import math
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from model import DeepAcr


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
onehot_index = {'A': 0,'V': 1, 'G': 2, 'I': 3, 'L': 4, 'F': 5, 'P':6, 'Y':7,
                'M': 8,'T': 9, 'S': 10, 'H': 11, 'N': 12, 'Q': 13, 'W': 14,
                'R': 15, 'K': 16, 'D': 17, 'E': 18, 'C':19}
ss3_index = {'C': 0, 'E': 1, 'H': 2}
ss8_index = {'L': 0, 'H': 1, 'T': 2, 'E': 3, 'S': 4, 'G': 5, 'B': 6, 'I': 7}
acc_index = {'E': 0, 'B': 1, 'M': 2}


def read_ss_data(file_path):
    seq = []
    ss3 = []
    ss8 = []
    acc = []
    for index in sorted(os.listdir(file_path)):
        if os.path.isdir(os.path.join(file_path, index)):
            f = open(file_path+'/'+index+'/'+index+'.ss3_simp.txt', 'r')
            lines = f.read().splitlines()
            seq.append([onehot_index[char] for char in lines[1]])
            ss3.append([ss3_index[char] for char in lines[2]])
            seq[-1] = torch.tensor(seq[-1])
            ss3[-1] = torch.tensor(ss3[-1])
            f.close()

            f = open(file_path+'/'+index+'/'+index+'.ss8_simp.txt', 'r')
            lines = f.read().splitlines()
            ss8.append([ss8_index[char] for char in lines[2]])
            ss8[-1] = torch.tensor(ss8[-1])
            f.close()

            f = open(file_path+'/'+index+'/'+index+'.acc_simp.txt', 'r')
            lines = f.read().splitlines()
            acc.append([acc_index[char] for char in lines[2]])
            acc[-1] = torch.tensor(acc[-1])
            f.close()

    return copy.deepcopy(seq), copy.deepcopy(ss3), copy.deepcopy(ss8), copy.deepcopy(acc)

def data_2_onehot(data, classes):
    data = pad_sequence(data, batch_first=True)
    data = torch.unsqueeze(data, 1)
    data = F.one_hot(data, num_classes=classes).float()
    return copy.deepcopy(data).to(device)

def read_pssm_data(file_path):
    dpc_f = np.loadtxt(file_path+'/dpc_pssm.csv', delimiter=",", dtype='float32')
    ac_f = np.loadtxt(file_path+'/pssm_ac.csv', delimiter=",", dtype='float32')
    com_f = np.loadtxt(file_path+'/pssm_composition.csv', delimiter=",", dtype='float32')
    r_f = np.loadtxt(file_path+'/rpssm.csv', delimiter=",", dtype='float32')

    pssm = np.hstack((dpc_f, ac_f, com_f, r_f))
    
    return pssm


if __name__ == '__main__':
    ''' seq, ss3, ss8, acc '''
    test_pos_seq, test_pos_ss3, test_pos_ss8, test_pos_acc = read_ss_data('data/pos/ss')
    test_neg_seq, test_neg_ss3, test_neg_ss8, test_neg_acc = read_ss_data('data/neg/ss')

    seq = [*test_pos_seq, *test_neg_seq]
    ss8 = [*test_pos_ss8, *test_neg_ss8]
    ss3 = [*test_pos_ss3, *test_neg_ss3]
    acc = [*test_pos_acc, *test_neg_acc]

    seq = data_2_onehot(seq, 20)
    ss3 = data_2_onehot(ss3, 3)
    ss8 = data_2_onehot(ss8, 8)
    acc = data_2_onehot(acc, 3)

    ''' 4 PSSM features '''
    test_pos_pssm = read_pssm_data('data/pos/pssm')
    test_neg_pssm = read_pssm_data('data/neg/pssm')

    dnn_data = []
    label = []
    ''' Transformer features '''
    for i in range(10):
        label.append(1)
        embs = torch.load('data/pos/trans/'+str(i)+'.pt')
        dnn_data.append(np.append(test_pos_pssm[i], embs['mean_representations'][33]))
    for i in range(10):
        label.append(0)
        embs = torch.load('data/neg/trans/'+str(i)+'.pt')
        dnn_data.append(np.append(test_neg_pssm[i], embs['mean_representations'][33]))

    dnn_data = torch.tensor(dnn_data).to(device)
    label = np.array(label)

    print("finish preparing data")

    model = DeepAcr()
    model.load_state_dict(torch.load('./model.ckpt'))
    model.to(device)
    pred = model(seq, ss3, ss8, acc, dnn_data)
    pred = pred.cpu().detach()
    pred = pred.argmax(dim=-1)

    C = confusion_matrix(label, pred, labels=[0, 1])
    TN = C[0][0]
    FN = C[0][1]
    FP = C[1][0]
    TP = C[1][1]
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + FP + TN + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F_value = 2 * TP / (2 * TP + FP + FN)
    MCC = (TP * TN - FN * FP) / math.sqrt((TP + FN) * (TN + FP) * (TP + FP) * (TN + FN)) 
    print("TN: {:.4f}, FN: {:.4f}, FP: {:.4f}, TP: {:.4f}".format(TN, FN, FP, TP))
    print("SN: {:.4f}, SP: {:.4f}, ACC: {:.4f}, Precision: {:.4f}, Recall: {:.4f} F-value: {:.4f}, MCC: {:.4f}".\
            format(SN, SP, ACC, Precision, Recall, F_value, MCC))

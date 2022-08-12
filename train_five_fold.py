import os
import copy
import math
import random
import numpy as np
from Bio import SeqIO
from operator import itemgetter
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from model import AcrNET


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seq_index = {'A': 0,'V': 1, 'G': 2, 'I': 3, 'L': 4, 'F': 5, 'P':6, 'Y':7,
                'M': 8,'T': 9, 'S': 10, 'H': 11, 'N': 12, 'Q': 13, 'W': 14,
                'R': 15, 'K': 16, 'D': 17, 'E': 18, 'C':19}
ss3_index = {'C': 0, 'E': 1, 'H': 2}
ss8_index = {'L': 0, 'H': 1, 'T': 2, 'E': 3, 'S': 4, 'G': 5, 'B': 6, 'I': 7}
acc_index = {'E': 0, 'B': 1, 'M': 2}

def read_seq(file_path):
    seq = []
    for record in SeqIO.parse(file_path,'fasta'):
        seq.append(torch.tensor([seq_index[char] for char in record.seq]))
    return seq

def read_raptorx(file_path):
    ss3 = []
    ss8 = []
    acc = []

    f = open(file_path+'ss3.txt', 'r')
    lines = f.read().splitlines()
    for l in lines:
        ss3.append(torch.tensor([ss3_index[char] for char in l]))
    f.close()

    f = open(file_path+'ss8.txt', 'r')
    lines = f.read().splitlines()
    for l in lines:
        ss8.append(torch.tensor([ss8_index[char] for char in l]))
    f.close()

    f = open(file_path+'acc.txt', 'r')
    lines = f.read().splitlines()
    for l in lines:
        acc.append(torch.tensor([acc_index[char] for char in l]))
    f.close()

    return ss3, ss8, acc

def data_2_onehot(data, classes):
    data = pad_sequence(data, batch_first=True)
    data = torch.unsqueeze(data, 1)
    data = F.one_hot(data, num_classes=classes).float().to(device)
    return copy.deepcopy(data)

def read_pssm(file_path):
    pssm = np.loadtxt(file_path+'pssm.csv', delimiter=",", dtype='float32')
    pssm = torch.from_numpy(pssm)
    
    return pssm

def train(seq, ss3, ss8, acc, data, label, train_index, test_index):
    seq_test = list(itemgetter(*test_index)(seq))
    ss3_test = list(itemgetter(*test_index)(ss3))
    ss8_test = list(itemgetter(*test_index)(ss8))
    acc_test = list(itemgetter(*test_index)(acc))

    data_test = data[test_index]
    data_test = torch.tensor(data_test).to(device)
    label_test = label[test_index]
    label_test = torch.tensor(label_test)

    ''' prepare test data '''
    seq_test = data_2_onehot(seq_test, 20)
    ss3_test = data_2_onehot(ss3_test, 3)
    ss8_test = data_2_onehot(ss8_test, 8)
    acc_test = data_2_onehot(acc_test, 3)

    '''train and eval loop'''
    for step in range(3001):
        model.train()

        batch_index = np.random.choice(train_index, batch_size)

        seq_batch = [seq[i] for i in batch_index]
        batch_seq = data_2_onehot(seq_batch, 20)

        ss3_batch = [ss3[i] for i in batch_index]
        batch_ss3 = data_2_onehot(ss3_batch, 3)

        ss8_batch = [ss8[i] for i in batch_index]
        batch_ss8 = data_2_onehot(ss8_batch, 8)

        acc_batch = [acc[i] for i in batch_index]
        batch_acc = data_2_onehot(acc_batch, 3)


        pssm_trans_batch = data[batch_index]
        pssm_trans_batch = torch.from_numpy(pssm_trans_batch).to(device)

        label_batch = label[batch_index]
        label_batch = torch.tensor(label_batch).to(device)

        pred = model(batch_seq, batch_ss3, batch_ss8, batch_acc, pssm_trans_batch)
        loss = loss_function(pred, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 3000:
            pred = pred.argmax(dim=-1)
            train_acc = torch.mean((pred == label_batch).type(torch.float)).item()
            train_loss = loss.item()
            print("train, step {}, acc {:.4f}, loss {:.4f}".format(step, train_acc, train_loss))

            # eval
            model.eval()

            pred = model(seq_test, ss3_test, ss8_test, acc_test, data_test)
            pred = pred.cpu().detach()
            pred = pred.argmax(dim=-1)
            acc = torch.mean((pred == label_test).type(torch.float)).item()
            C = confusion_matrix(label_test, pred, labels=[0, 1])
            TN = C[0][0]
            FN = C[0][1]
            FP = C[1][0]
            TP = C[1][1]
            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            ACC = (TP + TN) / (TP + FP + TN + FN)
            F_value = 2 * TP / (2 * TP + FP + FN)
            MCC = (TP * TN - FN * FP) / math.sqrt((TP + FN) * (TN + FP) * (TP + FP) * (TN + FN)) 
            print("TN: {:.4f}, FN: {:.4f}, FP: {:.4f}, TP: {:.4f}".format(TN, FN, FP, TP))
            print("Precision: {:.4f}, Recall: {:.4f}, ACC: {:.4f}, F-value: {:.4f}, MCC: {:.4f}".format(Precision, Recall, ACC, F_value, MCC))

            return Precision, Recall, ACC, F_value, MCC

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--data_path", dest="data_path",
                    help="path of data folder", default = "./data/")
    args = parser.parse_args()

    ''' seq '''
    pos_seq = read_seq(args.data_path+'/fastas/pos_tr_te_indices.fasta')
    neg_seq = read_seq(args.data_path+'/fastas/neg_tr_te_indices.fasta')
    seq = [*pos_seq, *neg_seq]

    pos_size = len(pos_seq)
    neg_size = len(neg_seq)

    ''' ss3, ss8, acc '''
    pos_ss3, pos_ss8, pos_acc = read_raptorx(args.data_path+'/RaptorX/pos/')
    neg_ss3, neg_ss8, neg_acc = read_raptorx(args.data_path+'/RaptorX/neg/')
    ss3 = [*pos_ss3, *neg_ss3]
    ss8 = [*pos_ss8, *neg_ss8]
    acc = [*pos_acc, *neg_acc]

    ''' 4 PSSM features '''
    pos_pssm = read_pssm(args.data_path+'/POSSUM/pos/')
    neg_pssm = read_pssm(args.data_path+'/POSSUM/neg/')

    dnn_data = []

    ''' ESM features '''
    for i in range(pos_size):
        embs = torch.load(args.data_path+'/ESM-1b/pos/'+str(i)+'.pt')
        dnn_data.append(np.append(pos_pssm[i], embs['mean_representations'][33]))
    for i in range(neg_size):
        embs = torch.load(args.data_path+'/ESM-1b/neg/'+str(i)+'.pt')
        dnn_data.append(np.append(neg_pssm[i], embs['mean_representations'][33]))
    dnn_data = np.array(dnn_data)

    label = [1]*pos_size + [0]*neg_size
    label = np.array(label)

    kf = KFold(n_splits=5, shuffle=True)

    acc_sum = 0.0
    mcc_sum = 0.0
    pre_sum = 0.0
    rec_sum = 0.0
    f_sum = 0.0

    i = 0
    for kl_train_index, kl_test_index in kf.split(label):
        print()
        print("===============================Round "+str(i)+"===============================")
        i += 1
        ''' model and optimizer and loss function '''
        model = AcrNET()
        model.to(device)
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        Precision, Recall, ACC, F_value, MCC = train(seq, ss3, ss8, acc, dnn_data, label, kl_train_index, kl_test_index)

        acc_sum += ACC
        mcc_sum += MCC
        pre_sum += Precision
        rec_sum += Recall
        f_sum += F_value

    print("precision, {:.4f}".format(pre_sum/5))
    print("recall, {:.4f}".format(rec_sum/5))
    print("acc, {:.4f}".format(acc_sum/5))
    print("mcc, {:.4f}".format(mcc_sum/5))
    print("f-vale, {:.4f}".format(f_sum/5))

import torch
import copy
import pickle
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

use_cuda = torch.cuda.is_available()


def custom_collate_fn(batch):
    # input is a list of dialogturn objects
    bt_siz = len(batch)
    # sequence length only affects the memory requirement, otherwise longer is better
    pad_idx, max_seq_len = 52700, 60
    
   

    maxturnsnumber = 0
    minturn = 100
    turnsnumbers = []
    for i, (d, cl_u) in enumerate(batch):
        turnsnumber = len(cl_u)
        if turnsnumber >= maxturnsnumber:
            maxturnsnumber = turnsnumber
        if turnsnumber <= minturn:
            minturn = turnsnumber
    u_batch = []
    u_lens = []
    l_u = []
    max_clu = 0
    for j in range (0, maxturnsnumber):
        u_lensj = np.zeros(bt_siz, dtype = int)
        u_lens.append(u_lensj)
        l_u.append(0)
        u_batchj = []
        for i in range(0, bt_siz):
           u_batchj.append(torch.LongTensor([52700]))
        u_batch.append(u_batchj)
    for i, (d, cl_u)in enumerate(batch):
        turnsnumbers.append(len(cl_u))
        for j,(cl_uj) in enumerate(cl_u):
           cl_u[j] = min(cl_uj, max_seq_len)
           if cl_u[j] > l_u[j]:
               l_u[j] = cl_u[j]
           if cl_u[j] > max_clu:    
               max_clu = cl_u[j]
               
           u_batch[j][i] = torch.LongTensor(d.u[j])
           u_lens[j][i] = cl_u[j]
    t = u_batch.copy()
    for j in range(0, maxturnsnumber):
        u_batch[j] = Variable(torch.ones( bt_siz, max_clu).long() * pad_idx)
    end_tok = torch.LongTensor([2])
    for i in range(bt_siz):
        for j in range (0, maxturnsnumber):
            seq, cur_l = t[j][i], t[j][i].size(0)
       
            if cur_l <= max_clu:
                u_batch[j][i, :cur_l].data.copy_(seq[:cur_l])
            else:
                u_batch[j][i, :].data.copy_(torch.cat((seq[:l_u[j]-1],end_tok),0))      
    sort4utterlength = []
    #for j in range(0, maxturnsnumber):
    #    sort4utterlength.append(np.argsort(u_lens[j]*-1))
    #    u_batch[j] = u_batch[j][sort4utterlength[j], :]
    #    u_batch[j] = np.array(u_batch[j])
    #    u_lens[j] = u_lens[j][sort4utterlength[j]
    #print(u_batch)
    #u_batch = np.array(u_batch)
    #sort4utternumber = np.argsort(turnsnumber*-1)
    #u_batch = u_batch[sort4utternumber,:
    return u_batch, u_lens, turnsnumbers, max_clu, minturn

class DialogTurn:
    def __init__(self, item):
        self.u = []
        cur_list  = []
        max_word = 0
        for d in item:
            cur_list.append(d)
            if d == 1:
                self.u.append(copy.copy(cur_list))
                cur_list = []
            if d > max_word:
                max_word = d
    def __len__(self):
        length = 0
        for utter in self.u:
            length += len(utter)
        return length

    def __repr__(self):
        strin = ""
        for utter in self.u:
            strin += str(utter)
        return strin


class MovieTriples(Dataset):
    def __init__(self, data_type, length=None):
        if data_type == 'train':
            _file = 'train_sorted.dialogues.pkl'
        elif data_type == 'valid':
            _file = 'valid_sorted.dialogues.pkl'
        elif data_type == 'test':
            _file = 'test_sorted.dialogues.pkl'
        self.utterance_data = []

        with open(_file, 'rb') as fp:
            data = pickle.load(fp)
            for d in data:
                self.utterance_data.append(DialogTurn(d))
        # it helps in optimization that the batch be diverse, definitely helps!
        # self.utterance_data.sort(key=cmp_to_key(cmp_dialog))
        if length:
            self.utterance_data = self.utterance_data[2000:2000 + length]

    def __len__(self):
        return len(self.utterance_data)

    def __getitem__(self, idx):
        dialog = self.utterance_data[idx]
        length = []
        for utter in dialog.u:
            length.append(len(utter))
        return dialog, length


def tensor_to_sent(x, inv_dict, greedy=False):
    sents = []
    inv_dict[52700] = '<pad>'
    for li in x:
        if not greedy:
            scr = li[1]
            seq = li[0]
        else:
            scr = 0
            seq = li
        sent = []
        seq = seq[1:]
        for i in seq:
            i = i.item()
            sent.append(inv_dict[i])
            if i == 1:
                break
        sents.append((" ".join(sent), scr))
    return sents

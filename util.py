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
    #u_batch = []
    #u_lens = []
    #t = []
    #u1_biatch, u2_batch, u3_batch = [], [], []
    #u1_lens, u2_lens, u3_lens = np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int)

    # these store the max sequence lengths for the batch
    #l_u1, l_u2, l_u3 = 0, 0, 0
        
    #l_u = []
    #find the maximum turns
    #for i, (d, cl_u) in enumerate(batch):
    #    for cl_utter in cl_u:
    #        cl_utter = min(cl_utter, max_seq_len)
    #        l_utter = 0
    #        if cl_utter > l_utter:
    #            l_utter = cl_utter
    #        utter_lens = np.zeros(bt_siz, dtype = int)
    #        utter_lens
    #        l_u.append(l_utter)
    #    for j, (utter) in enumerate(d.u):
    #        utter_batch = []
    #        utter_batch.append(torch.LongTensor(utter))
    #        t.append(utter_batch)
    #        utter_batch = Variable(torch.ones(bt_siz, l_u[j]).long * pad_idx)
    #        u_batch.append(utter_batch)
    #       end_tok = torch.LongTensor([2])

    #for i in range(bt_siz):
    #    for j,turn in t:
    #        seq, cur_l = turn[i], turn[i].size()
    #        if cur_l <= l_u[j]:
    #            u_batch[i, :cul_l][j].data.copy_(seq[:cur_l]) 
    #        else: 
    #            u_batch[i, :][j].data.copy_(torch.cat((seq[:l_u[j]-1], end_tok),0))

    # for i, (d, cl_u1, cl_u2, cl_u3) in enumerate(batch):
    #    cl_u1 = min(cl_u1, max_seq_len)
    #    cl_u2 = min(cl_u2, max_seq_len)
    #    cl_u3 = min(cl_u3, max_seq_len)

    #    if cl_u1 > l_u1:
    #        l_u1 = cl_u1
    #    u1_batch.append(torch.LongTensor(d.u1))
    #    u1_lens[i] = cl_u1

    #    if cl_u2 > l_u2:
    #        l_u2 = cl_u2
    #    u2_batch.append(torch.LongTensor(d.u2))
    #    u2_lens[i] = cl_u2

    #    if cl_u3 > l_u3:
    #        l_u3 = cl_u3
    #    u3_batch.append(torch.LongTensor(d.u3))
    #    u3_lens[i] = cl_u3

    # t1, t2, t3 = u1_batch, u2_batch, u3_batch

    #u1_batch = Variable(torch.ones(bt_siz, l_u1).long() * pad_idx)
    #u2_batch = Variable(torch.ones(bt_siz, l_u2).long() * pad_idx)
    #u3_batch = Variable(torch.ones(bt_siz, l_u3).long() * pad_idx)
    #end_tok = torch.LongTensor([2])

    #for i in range(bt_siz):
    #    seq1, cur1_l = t1[i], t1[i].size(0)
    #    if cur1_l <= l_u1:
    #        u1_batch[i, :cur1_l].data.copy_(seq1[:cur1_l])
    #    else:
    #        u1_batch[i, :].data.copy_(torch.cat((seq1[:l_u1-1], end_tok), 0))

    #    seq2, cur2_l = t2[i], t2[i].size(0)
    #    if cur2_l <= l_u2:
    #        u2_batch[i, :cur2_l].data.copy_(seq2[:cur2_l])
    #    else:
    #        u2_batch[i, :].data.copy_(torch.cat((seq2[:l_u2-1], end_tok), 0))

    #    seq3, cur3_l = t3[i], t3[i].size(0)
    #    if cur3_l <= l_u3:
    #        u3_batch[i, :cur3_l].data.copy_(seq3[:cur3_l])
    #    else:
    #        u3_batch[i, :].data.copy_(torch.cat((seq3[:l_u3-1], end_tok), 0))

    #sort1, sort2, sort3 = np.argsort(u1_lens*-1), np.argsort(u2_lens*-1), np.argsort(u3_lens*-1)
    # cant call use_cuda here because this function block is used in threading calls

    #return u1_batch[sort1, :], u1_lens[sort1], u2_batch[sort2, :], u2_lens[sort2], u3_batch[sort3, :], u3_lens[sort3]
    maxturnsnumber = 0
    turnsnumbers = []
    for i, (d, cl_u) in enumerate(batch):
        turnsnumber = len(cl_u)
        if turnsnumber >= maxturnsnumber:
            maxturnsnumber = turnsnumber
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
    #u_batch = u_batch[sort4utternumber]
    return u_batch, u_lens, turnsnumbers, max_clu

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
            _file = 'train.dialogues.pkl'
        elif data_type == 'valid':
            _file = 'valid.dialogues.pkl'
        elif data_type == 'test':
            _file = 'test.dialogues.pkl'
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
        for i in seq:
            sent.append(inv_dict[i])
            if i == 2:
                break
        sents.append((" ".join(sent), scr))
    return sents

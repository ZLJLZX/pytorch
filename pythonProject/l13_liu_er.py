import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import gzip
import csv
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


HIDDEN_SIZE = 100
BATCH_SIZE =256
N_LAYER =2
N_EPOCHS =100
N_CHARS =128
USE_GPU = False


res = []
is_train_set = True




class NameDateset(Dataset):
    def __init__(self, is_train_set=True):
        filename = 'names_train.csv.gz' if is_train_set else 'names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.country = [row[1] for row in rows]
        self.country_list = list(set(self.country))
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)
    def __getitem__(self, item):
        return self.names[item], self.country_dict[self.country[item]]
    def __len__(self):
        return self.len
    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list,0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self,index):
        return self.country_list[index]

    def getCountryNum(self):
        return self.country_num


train_set = NameDateset(is_train_set=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_set = NameDateset(is_train_set=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = train_set.getCountryNum()




class RNNClassifier(torch.nn.Module):
    def __init__(self,input_size, hidden_size,output_size,n_layers=1,bidirectional = True):
        super(RNNClassifier,self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.embedding = torch.nn.Embedding(input_size,hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers,bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size*self.n_directions,output_size)


    # def create_tensor(self, countries):
    #     sequences_and_lengths = [name2list(name) for name in names]
    #     name_sequences = [sl[0] for sl in sequences_and_lengths]
    #     seq_lengths = torch.LongTensor([sl[1]for sl in sequences_and_lengths])
    #     countries = countries.long()
    #
    #
    #     seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    #     for idx, (seq,seq_len) in enumerate(zip(name_sequences, name_lengths), 0):
    #         seq_tensor[idx,:seq_len] = torch.LongTensor(seq)
    #
    #     seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending = True)
    #     seq_tensor = seq_tensor[perm_idx]
    #     countries = countries[perm_idx]
    #
    #     return create_tensor(seq_tensor),create_tensor(seq_lengths),create_tensor(countries)
    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        gru_input = pack_padded_sequence(embedding,seq_lengths)

        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions ==2:
            hidden_cat = torch.cat([hidden[-1],hidden[-2]],dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)

        return fc_output

def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]  # 每一个名字都变成ASCII列表
    name_sequences = [sl[0] for sl in sequences_and_lengths]  # 因为name2list既返回了名字的列表也返回了名字的长度
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()  # 从数据集里面拿出来的countries就是一个整数，将其转变为long

    # 接下来是做padding的过程，先做一个全0的张量，然后再把原先的名字张量粘贴过去
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # 排序(按照序列的长度)
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)  # 返回排序后的长度和索引
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]
    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)

def name2list(name):  # 读出每个字符的ASCII码值
    arr = [ord(c) for c in name]
    return arr, len(arr)
def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "{}m {:.0f}s".format(m, s)



def trainModel():
    total_loss = 0
    for i, (names,countries) in enumerate(train_loader,1):
        input, seq_lengths, target = make_tensors(names, countries)
        output = classifier(input, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{i * len(input)} / {len(train_set)}]', end='')
            print(f'loss = {total_loss/(i*len(input))}')
    return total_loss

def testModel():
    correct = 0
    total = len(test_set)
    print("evaluating trained model```")
    with torch.no_grad():
        for i, (names, countries) in enumerate(test_loader,1):
            input, seq_lengths, target = make_tensors(names,countries)
            output = classifier(input,seq_lengths)
            pred = output.max(dim= 1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' %(100* correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct/total










if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS,HIDDEN_SIZE,N_COUNTRY,N_LAYER)
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Rprop(classifier.parameters(), lr=0.01)

    start = time.time()
    print("Training for %d epochs```" % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        trainModel()
        acc = testModel()
        acc_list.append(acc)

epoch = np.arange(1, len(acc_list)+1, 1)
acc_list = np.array(acc_list)
plt.plot(epoch, acc_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()


torch.save(classifier.state_dict(), 'name_classifier_model.pt')

# import torch
#
# batch_size = 1
# seq_len = 3
# input_size = 4
# hidden_size = 2
# num_layers = 1
#
# cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#
#
# inputs = torch.randn(batch_size, seq_len, input_size)
# hidden = torch.zeros(num_layers, batch_size,hidden_size)
#
# out,hidden = cell(inputs, hidden)
#
# print('Output size:', out.shape)
# print('Output', out)
# print('Hidden size:', hidden.shape)
# print('Hidden:', hidden)
# # for idx, input in enumerate(dataset):
#     print('=' * 20, idx, '=', * 20)
#     print('input size', input.shape)
#
#     hidden = cell(input,hidden)
#
#     print('outputs size:' , hidden.shape)
#     print(hidden)
import matplotlib.pyplot as plt
import torch

batch_size = 1
seq_len = 5
hidden_size = 4
input_size = 4
num_layers = 1

idx2char = ['e','h','l','o']
x_data = [1,0,2,2,3]
y_data = [3,1,2,3,2]

one_hot_lookup = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(seq_len,batch_size,input_size)
labels = torch.LongTensor(y_data)   #.view(-1,1)


class Model(torch.nn.Module):
    def __init__(self,input_size, hidden_size, batch_size,num_layers):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self,input):
        hidden = torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        out,_ = self.rnn(input,hidden)
        return out.view(-1,self.hidden_size)

    # def init_hidden(self):
    #     return torch.zeros(self.batch_size,self.hidden_size)

net = Model(input_size,hidden_size,batch_size,num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.01)

if __name__ == '__main__':
    epoch_list = []
    loss_list = []

    for epoch in range(200):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=1)
        idx = idx.data.numpy()
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(',Epoch [%d/20] loss=%.3f' % (epoch + 1, loss.item()))
        epoch_list.append(epoch)
        loss_list.append(loss.item())

plt.plot(epoch_list,loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# for epoch in range(15):
#     loss = 0
#     optimizer.zero_grad()
#
#     print('Predicted String:', end='')
#     for input, label in zip(inputs,labels):
#         hidden = net(input)
#         loss += criterion(hidden,label)
#         _, idx = hidden.max(dim=1)
#         print(idx2char[idx.item()], end='')
#     loss.backward()
#     optimizer.step()
#     print(', Epoch [%d/15] loss=%.4f' % (epoch+1, loss.item()))
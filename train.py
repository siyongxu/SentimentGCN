import pickle
import time
import numpy as np
from utils import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from models import GCN


dataset = 'ratings'

# Load data
with open("data/ind.{}.adj".format(dataset), 'rb') as f:
    adj = pickle.load(f)

with open("data/ind.{}.data".format(dataset), 'rb') as f:
    data = pickle.load(f)

num_doc = data['num_doc']
A = np.array(adj.toarray())

features = sp.eye(A.shape[0])
labels = torch.LongTensor(np.array(data['label']))
features = torch.FloatTensor(np.array(features.todense()))
adj = sparse_mx_to_torch_sparse_tensor(adj)
idx_train = torch.LongTensor(range(data['train_size']))
idx_val = torch.LongTensor(range(data['val_size']))
idx_test = torch.LongTensor(range(data['test_size']))

#Hyperparameter
epochs = 200
hidden = 200
dropout = 0.5
lr = 0.02
weight_decay = 0
cuda = 0
early_stopping = 10
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=hidden,
            nclass=2,
            dropout=dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=lr, weight_decay=weight_decay)

if cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

cost_val = []

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    model.eval()
    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    cost_val.append(loss_val)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    if epoch > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping + 1):-1]):
        print("Early stopping...")
        return False
    return True

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(epochs):
    flag = train(epoch)
    if flag == False :
        break
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
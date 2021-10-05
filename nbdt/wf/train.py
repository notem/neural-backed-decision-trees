import os
import argparse
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from nbdt.data.wf import Pylls, Subpages
from nbdt.models.df import _DFNet as model


device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("==> Preparing data..")

input_size = 9000
#trainset = Subpages(
trainset = Pylls(
    root="../../data/wf-spring",
    #root="../data/subpages24x90x90",
    length=input_size,
    train=True,
    download=True,
    include_unm = True,
    defen_multiples=20,
)
#testset = Subpages(
testset = Pylls(
    root="../../data/wf-spring",
    #root="../data/subpages24x90x90",
    length=input_size,
    train=False,
    download=True,
    include_unm = True,
    defen_multiples=20,
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4
)

net = model(num_classes=len(trainset.classes), input_size=input_size)
net = net.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint_fname = 'DFNet-Subpages'
checkpoint_path = "./checkpoint/{}.pth".format(checkpoint_fname)
print(f"==> Checkpoints will be saved to: {checkpoint_path}")


#optimizer = optim.SGD(net.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adamax(net.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()


for epoch in range(90):
    net.train()
    train_loss = 0.
    train_acc = 0.
    n = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, y_pred = torch.max(outputs, 1)
        train_acc += torch.sum(y_pred == targets)
        n += len(targets)
    train_loss /= batch_idx + 1
    train_acc /= n
    print(f'[{epoch}] tr. loss ({train_loss:0.3f}), tr. acc ({train_acc:0.3f}),', end='', flush=True)


    net.eval()
    test_loss = 0.
    test_acc = 0.
    n = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, y_pred = torch.max(outputs, 1)
        test_acc += torch.sum(y_pred == targets)
        n += len(targets)
    test_loss /= batch_idx + 1
    test_acc /= n
    print(f' te. loss ({test_loss:0.3f}), te. acc ({test_acc:0.3f})')

    torch.save(net.state_dict(), checkpoint_path)

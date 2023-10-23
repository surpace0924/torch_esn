#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import esn

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


torch_fix_seed() 


import matplotlib.pyplot as plt
def save_plot(model, UT_test, DT_test, i):
    device = torch.device('cuda')
    UT = torch.from_numpy(UT_test).to(device)
    UT = torch.unsqueeze(UT, dim=-1)
    y, _ = model(UT)


    fig = plt.figure()
    plt.ylim([0, 1.5])
    plt.plot(DT_test[:400])
    plt.plot(y.to('cpu').detach().numpy().copy()[:400])
    plt.show()
    # plt.savefig(f'{str(i).zfill(4)}.png')


def main():
    # data = np.sin(np.arange(1000)/10).astype(np.float32)
    step = 5
    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float32).T[0]

    train_len = int(len(data) * 0.8)
    test_len = len(data) - train_len - step

    # [T, N_u]
    UT_train = data[:train_len]
    DT_train = data[step:train_len+step]
    UT_test = data[train_len:train_len+test_len]
    DT_test = data[train_len+step:]

    model = esn.ESN(1, 8, 1)
    device = torch.device('cuda')
    model.to(device=device)

    UT = torch.from_numpy(UT_train).to(device)
    DT = torch.from_numpy(DT_train).to(device)
    UT = torch.unsqueeze(UT, dim=-1)
    DT = torch.unsqueeze(DT, dim=-1)

    # for param in esn.parameters():
    #     print(param)
    # print()
    
    # 逆行列による最適化
    model(UT, 800, DT)
    model.fit()

    # # 勾配法による最適化
    # epoch_num = 100
    # criterion = nn.MSELoss()
    # import torch.optim as optim
    # optimizer = torch.optim.Adam(esn.parameters(), lr=1e-2)
    # for epoch in range(epoch_num):
    #     preds, _ = esn(UT)
    #     loss = criterion(preds, DT)
    #     print(loss)
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     save_plot(esn, UT_test, DT_test, epoch)

    # for param in esn.parameters():
    #     print(param)
    
    save_plot(model, UT_test, DT_test, 100)
    

if __name__ == '__main__':
    main()

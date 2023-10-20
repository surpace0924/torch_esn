#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

import random

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


class ESN(nn.Module):
    def __init__(self,
                 N_u,
                 N_x,
                 N_y,
                 density=0.05,  
                 input_scale=1.0,
                 rho=0.95,
                 leaking_rate=1.0,
                 regularization_rate=1.0):
        super().__init__()
        # 各ベクトルの次元数
        self.N_u = N_u
        self.N_x = N_x
        self.N_y = N_y
        
        # 重み行列の定義
        W_in  = torch.Tensor(N_x, N_u).uniform_(-input_scale, input_scale)
        W     = self.make_W(N_x, density, rho)
        W_out = torch.zeros(N_y, N_x)

        # モデルのパラメータ登録
        # W_out以外は重み更新を禁止する
        self.W_in  = nn.Parameter(W_in,  requires_grad=False)
        self.W     = nn.Parameter(W,     requires_grad=False)
        self.W_out = nn.Parameter(W_out, requires_grad=True)
        
        # リザバー状態ベクトルと逆行列計算用行列
        self.x = torch.Tensor(N_x)
        self.D_XT = torch.Tensor(N_y, N_x)    # [N_y, N_x]
        self.X_XT = torch.Tensor(N_x, N_x)    # [N_x, N_x]

        # LIの漏れ率とリッジ回帰の正則化係数
        self.alpha = leaking_rate
        self.beta = regularization_rate
        

    # density: 結合密度
    # density: スペクトル半径
    def make_W(self, N_x, density, spectral_radius):
        # N_x*N_x次元のベクトルを用意
        W = torch.Tensor(N_x * N_x)

        # [-1.0, 1.0] の乱数で初期化
        W.uniform_(-1.0, 1.0)

        # 結合密度に応じて W の要素を 0 にする
        zero_idx = torch.randperm(int(N_x * N_x))
        zero_idx = zero_idx[:int(N_x * N_x * (1 - density))]
        W[zero_idx] = 0.0

        # 行列形式にする
        W = W.view(N_x, N_x)

        # 指定したスペクトル半径となるようにリスケール
        eigs = torch.linalg.eigvals(W)
        max_eigs = torch.max(torch.abs(eigs))
        if max_eigs != 0:
            W = W * (spectral_radius / max_eigs)
        
        return W

    def reservoir(self, u, x, W_in, W, alpha):
        x = x.to(device=W.device)
        x = (1.0 - alpha) * x + alpha * torch.tanh(F.linear(u, W_in) + F.linear(x, W))
        return x

    # バッチ学習
    # U_T [T, N_u]
    # D_T [T, N_y]
    def fit(self):
        I = torch.eye(self.N_x)   # [N_x, N_x]
        beta_I = self.beta * I.to(torch.float32).to(device=self.W.device)
    
        # 出力重みの計算 [N_y, N_x]
        W_out = self.D_XT @ torch.inverse(self.X_XT + beta_I)
        self.W_out = nn.Parameter(W_out)


    # バッチ学習後の予測
    def forward(self, UT, trans_len = 0, DT = None):
        X, D, Y = [], [], []
        for n, u in enumerate(UT):
            # リザバーの時間発展と出力を計算
            self.x = self.reservoir(u, self.x, self.W_in, self.W, self.alpha)
            y = self.W_out @ self.x

            # 計算結果をappend
            X.append(torch.unsqueeze(self.x, dim=-1))
            Y.append(torch.unsqueeze(y, dim=-1))

            # 教師データがある場合はそれもappend
            if DT is not None:
                D.append(torch.unsqueeze(DT[n], dim=-1))
        
        # リザバー状態/出力ベクトルを横につなげた行列
        X = torch.cat(X, 1)     # [N_x, T-trans_len]
        Y = torch.cat(Y, 1)     # [N_y, T-trans_len]
        
        # 教師データがある場合は学習のための行列を計算
        if DT is not None:
            D = torch.cat(D, 1) # [N_y, T-trans_len]
            self.D_XT = D @ X.T # [N_y, N_x]
            self.X_XT = X @ X.T # [N_x, N_x]

        # 軸が逆のほうが扱いやすいため転置して返す
        return Y.T, X.T
    
import matplotlib.pyplot as plt
def save_plot(esn, UT_test, DT_test, i):
    device = torch.device('cuda')
    UT = torch.from_numpy(UT_test).to(device)
    UT = torch.unsqueeze(UT, dim=-1)
    y, _ = esn(UT)


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

    esn = ESN(1, 30, 1)
    device = torch.device('cuda')
    esn.to(device=device)

    UT = torch.from_numpy(UT_train).to(device)
    DT = torch.from_numpy(DT_train).to(device)
    UT = torch.unsqueeze(UT, dim=-1)
    DT = torch.unsqueeze(DT, dim=-1)

    # for param in esn.parameters():
    #     print(param)
    # print()
    
    # 逆行列による最適化
    esn(UT, 100, DT)
    esn.fit()

    # 勾配法による最適化
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
    #     # save_plot(esn, UT_test, DT_test, epoch)

    for param in esn.parameters():
        print(param)
    
    save_plot(esn, UT_test, DT_test, 100)
    

if __name__ == '__main__':
    main()

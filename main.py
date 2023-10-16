#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device('cuda')

import numpy as np
import networkx as nx


# エコーステートネットワーク
class ESN(nn.Module):
    # 各層の初期化
    def __init__(self,
                 N_u,
                 N_y,
                 N_x,
                 density=0.05,
                 input_scale=1.0,
                 rho=0.95,
                 leaking_rate=1.0):
        self.seed = 0
        np.random.seed(seed=self.seed)
        self.W_in = torch.Tensor(N_x, N_u).uniform_(-input_scale, input_scale).to(device)
        self.N_u = N_u

        self.W = self.make_W(N_x, density, rho)
        self.x = torch.Tensor(N_x).to(device)
        self.alpha = leaking_rate

        self.Wout = torch.Tensor(N_y, N_x).to(device)

        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x

    # density: 結合密度
    # density: スペクトル半径
    def make_W(self, N_x, density, spectral_radius):
        # N_x*N_x次元のベクトルを用意
        W = torch.Tensor(N_x * N_x).to(device)

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

    def reservoir(self, u, x, alpha, W_in, W):
        x = (1.0 - alpha) * x + alpha * torch.tanh(F.linear(u, W_in) + F.linear(x, W))
        return x

    # バッチ学習
    # U_T [T, N_u]
    # D_T [T, N_y]
    def train(self, UT, DT, trans_len = 10):
        train_len = len(UT)

        # 時間発展
        X, D= [], []
        for n, (u, d) in enumerate(zip(UT, DT)):
            # リザバー状態ベクトル
            self.x = self.reservoir(u, self.x, self.alpha, self.W_in, self.W)

            # 学習器
            if n >= trans_len:  # 過渡期を過ぎたら
                X.append(torch.unsqueeze(self.x, dim=-1))
                D.append(torch.unsqueeze(d, dim=-1))

        X = torch.cat(X, 1)                  # [N_x, T-trans_len]
        D = torch.cat(D, 1)                  # [N_y, T-trans_len]
        D_XT = D @ X.T                       # [N_y, N_x]
        X_XT = X @ X.T                       # [N_x, N_x]
        I = torch.eye(self.N_x).to(device)   # [N_x, N_x]
        beta_I = 0.001 * I.to(torch.float32) # [N_x, N_x]
        
        # 出力重みの計算 [N_y, N_x]
        self.Wout = D_XT @ torch.inverse(X_XT + beta_I)


    # バッチ学習後の予測
    def predict(self, UT):
        test_len = len(UT)
        Y_pred = []

        # 時間発展
        for n in range(test_len):
            u = UT[n]
            self.x = self.reservoir(u, self.x, self.alpha, self.W_in, self.W)

            # 学習後のモデル出力
            y_pred = self.Wout @ self.x
            Y_pred.append(y_pred)

        # モデル出力（学習後）
        return torch.cat(Y_pred)
    


import matplotlib.pyplot as plt

def main():
    # data = np.sin(np.arange(1000)/10).astype(np.float32)
    step = 10
    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float32).T[0]

    train_len = int(len(data) * 0.8)
    test_len = len(data) - train_len - step

    # [T, N_u]
    UT_train = data[:train_len]
    DT_train = data[step:train_len+step]
    UT_test = data[train_len:train_len+test_len]
    DT_test = data[train_len+step:]

    esn = ESN(1, 1, 5)

    UT = torch.from_numpy(UT_train).to(device)
    DT = torch.from_numpy(DT_train).to(device)
    UT = torch.unsqueeze(UT, dim=-1)
    DT = torch.unsqueeze(DT, dim=-1)
    esn.train(UT, DT)

    UT = torch.from_numpy(UT_test).to(device)
    UT = torch.unsqueeze(UT, dim=-1)
    y = esn.predict(UT)
    
    plt.plot(DT_test[:400])
    plt.plot(y.to('cpu').detach().numpy().copy()[:400])
    plt.show()

if __name__ == '__main__':
    main()

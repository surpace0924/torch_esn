#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.nn import functional as F
device = torch.device('cuda')

import numpy as np
import networkx as nx

# リッジ回帰（beta=0のときは線形回帰）
class Tikhonov:
    def __init__(self, N_x, N_y, beta):
        self.beta = beta
        self.X_XT = torch.zeros(N_x, N_x, dtype=torch.float32).to(device)
        self.D_XT = torch.zeros(N_y, N_x, dtype=torch.float32).to(device)
        self.N_y = N_y
        self.N_x = N_x

    # 学習用の行列の更新
    def __call__(self, d, x):
        x = torch.unsqueeze(x, dim=-1)
        d = torch.unsqueeze(d, dim=-1)
        self.X_XT += torch.matmul(x, x.T)
        self.D_XT += torch.matmul(d, x.T)

    # Woutの最適解（近似解）の導出
    def get_Wout_opt(self):
        I = np.identity(self.N_x).astype(np.float32)
        I = torch.from_numpy(I).to(device)
        X_pseudo_inv = torch.inverse(self.X_XT + self.beta*I)
        Wout_opt = torch.matmul(self.D_XT, X_pseudo_inv)
        return Wout_opt


# エコーステートネットワーク
class ESN:
    # 各層の初期化
    def __init__(self,
                 N_u,
                 N_y,
                 N_x,
                 density=0.05,
                 input_scale=1.0,
                 rho=0.95,
                 activation_func=np.tanh,
                 leaking_rate=1.0):
        self.seed = 0
        np.random.seed(seed=self.seed)
        self.W_in = np.random.uniform(-input_scale, input_scale, (N_x, N_u)).astype(np.float32)
        self.W_in = torch.from_numpy(self.W_in).to(device)
        self.N_u = N_u

        self.W = self.make_W(N_x, density, rho)
        self.x = np.zeros(N_x).astype(np.float32)  # リザバー状態ベクトルの初期化
        self.x = torch.from_numpy(self.x).to(device)
        self.activation_func = activation_func
        self.alpha = leaking_rate

        self.Wout = np.random.normal(size=(N_y, N_x)).astype(np.float32)
        self.Wout = torch.from_numpy(self.Wout).to(device)

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
    def train(self, U, D, optimizer, trans_len = 0):
        train_len = len(U)
        Y = []

        # 時間発展
        for n in range(train_len):
            u = U[n]
            d = D[n]
            
            # リザバー状態ベクトル
            self.x = self.reservoir(u, self.x, self.alpha, self.W_in, self.W)

            # 学習器
            if n > trans_len:  # 過渡期を過ぎたら
                optimizer(d, self.x)

            # 学習前のモデル出力
            y = torch.mv(self.Wout, self.x)
            Y.append(y)

        # 学習済みの出力結合重み行列を設定
        self.Wout = optimizer.get_Wout_opt()
        
        # モデル出力（学習前）
        return torch.cat(Y)

    # バッチ学習後の予測
    def predict(self, U):
        test_len = len(U)
        Y_pred = []

        # 時間発展
        for n in range(test_len):
            u = U[n]
            self.x = self.reservoir(u, self.x, self.alpha, self.W_in, self.W)

            # 学習後のモデル出力
            y_pred = torch.mv(self.Wout, self.x)
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
    U_train = data[:train_len]
    D_train = data[step:train_len+step]
    U_test = data[train_len:train_len+test_len]
    D_test = data[train_len+step:]

    esn = ESN(1, 1, 10)
    optimizer = Tikhonov(10, 1, 1e-3)

    U = torch.from_numpy(U_train).to(device)
    D = torch.from_numpy(D_train).to(device)
    U = torch.unsqueeze(U, dim=-1)
    D = torch.unsqueeze(D, dim=-1)
    esn.train(U, D, optimizer)

    U = torch.from_numpy(U_test).to(device)
    U = torch.unsqueeze(U, dim=-1)
    y = esn.predict(U)
    
    plt.plot(D_test[:400])
    plt.plot(y.to('cpu').detach().numpy().copy()[:400])
    plt.show()

if __name__ == '__main__':
    main()

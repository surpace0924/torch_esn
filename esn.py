#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

class ESN(nn.Module):
    def __init__(self,
                 N_u,
                 N_x,
                 N_y,
                 density=0.05,  
                 input_scale=1.0,
                 rho=0.95,
                 leaking_rate=0.95,
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
        self.x = torch.zeros(N_x)
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


    # U_T [T, N_u]
    # D_T [T, N_y]
    def fit(self):
        I = torch.eye(self.N_x)   # [N_x, N_x]
        beta_I = self.beta * I.to(torch.float32).to(device=self.W.device)
    
        # 出力重みの計算 [N_y, N_x]
        W_out = self.D_XT @ torch.inverse(self.X_XT + beta_I)
        self.W_out = nn.Parameter(W_out)


    def forward(self, UT, trans_len = 0, DT = None):
        X, Y = [], []
        for u in UT:
            # リザバーの時間発展と出力を計算
            self.x = self.reservoir(u, self.x, self.W_in, self.W, self.alpha)
            y = self.W_out @ self.x
            X.append(torch.unsqueeze(self.x, dim=-1))
            Y.append(torch.unsqueeze(y, dim=-1))

        # リザバー状態/出力ベクトルを横につなげた行列
        X = torch.cat(X, 1)     # [N_x, T]
        Y = torch.cat(Y, 1)     # [N_y, T]
        
        # 教師データがある場合は学習のための行列を計算
        if DT is not None:
            D = DT.T    # [N_y, T]
            cal_len = X.size()[1] - trans_len
            X_trimmed = X[:, :cal_len]       
            D_trimmed = D[:, :cal_len]
            self.D_XT = D_trimmed @ X_trimmed.T # [N_y, N_x]
            self.X_XT = X_trimmed @ X_trimmed.T # [N_x, N_x]

        # 軸が逆のほうが扱いやすいため転置して返す
        return Y.T, X.T
    
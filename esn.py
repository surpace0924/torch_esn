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
        self.N_l = 1
        
        # 重み行列の定義
        W_in  = torch.Tensor(N_x, N_u).uniform_(-input_scale, input_scale)
        W     = self.make_W(N_x, density, rho)
        W_out = torch.zeros(N_y, N_x)

        # モデルのパラメータ登録
        # W_out以外は重み更新を禁止する
        self.W_in  = nn.Parameter(W_in,  requires_grad=False)
        self.W     = nn.Parameter(W,     requires_grad=False)
        self.W_out = nn.Parameter(W_out, requires_grad=True)

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
        x = (1.0 - alpha) * x.t() + alpha * torch.tanh(F.linear(u.t(), W_in) + F.linear(x.t(), W))
        return x.t()


    # U_T [T, N_u]
    # D_T [T, N_y]
    def fit(self):
        I = torch.eye(self.N_x)   # [N_x, N_x]
        beta_I = self.beta * I.to(torch.float32).to(device=self.W.device)
    
        # 出力重みの計算 [N_y, N_x]
        W_out = self.D_XT @ torch.inverse(self.X_XT + beta_I)
        self.W_out = nn.Parameter(W_out)


    def forward(self, input, trans_len=0, target=None, x_0=None):
        device = self.W.device
        N_b = input.size()[1]
        
        # 初期状態の取得 or 初期化
        if x_0 is not None:
            self.x = x_0[0].t()
        else:
            self.x = torch.zeros(self.N_x, N_b).to(device)
        
        # 出力計算用行列
        self.D_XT = torch.zeros(self.N_y, self.N_x).to(device)
        self.X_XT = torch.zeros(self.N_x, self.N_x).to(device)

        # [T, N_b, N_u] -> [T, N_u, N_b]
        input = input.permute(0, 2, 1)
        if target is not None:
            target = target.permute(0, 2, 1)
        
        # 時間発展
        Y = []
        for n, u in enumerate(input):
            # リザバーの時間発展と出力を計算
            self.x = self.reservoir(u, self.x, self.W_in, self.W, self.alpha)
            y = F.linear(self.x.t(), self.W_out).t()
            Y.append(torch.unsqueeze(y, dim=0))

            # 過渡期を超えたら学習のための行列を計算
            if n >= trans_len and target is not None:
                d = target[n]
                self.D_XT += d @ self.x.t()
                self.X_XT += self.x @ self.x.t()

        output = torch.cat(Y, 0).permute(0, 2, 1) # [T, N_b, N_y]
        x_n = torch.unsqueeze(self.x.t(), dim=0)  # [N_l, N_b, N_x]
    
        return output, x_n
    
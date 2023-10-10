
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
device = torch.device('cuda')

import numpy as np
import networkx as nx


# 入力層
class Input:
    # 入力結合重み行列Winの初期化
    def __init__(self, N_u, N_x, input_scale, seed=0):
        '''
        param N_u: 入力次元
        param N_x: リザバーのノード数
        param input_scale: 入力スケーリング
        '''
        # 一様分布に従う乱数
        np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u)).astype(np.float32)
        self.Win = torch.from_numpy(self.Win).to(device)
        self.N_u = N_u

    # 入力結合重み行列Winによる重みづけ
    def __call__(self, u):
        '''
        param u: N_u次元のベクトル
        return: N_x次元のベクトル
        '''
        return torch.mv(self.Win, u)


# リザバー
class Reservoir:
    # リカレント結合重み行列Wの初期化
    def __init__(self, N_x, density, rho, activation_func, leaking_rate, seed=0):
        '''
        param N_x: リザバーのノード数
        param density: ネットワークの結合密度
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: ノードの活性化関数
        param leaking_rate: leaky integratorモデルのリーク率
        param seed: 乱数の種
        '''
        self.seed = seed
        self.W = self.make_connection(N_x, density, rho).astype(np.float32)
        self.W = torch.from_numpy(self.W).to(device)
        self.x = np.zeros(N_x).astype(np.float32)  # リザバー状態ベクトルの初期化
        self.x = torch.from_numpy(self.x).to(device)
        self.activation_func = activation_func
        self.alpha = leaking_rate

    # リカレント結合重み行列の生成
    def make_connection(self, N_x, density, rho):
        # Erdos-Renyiランダムグラフ
        m = int(N_x*(N_x-1)*density/2)  # 総結合数
        G = nx.gnm_random_graph(N_x, m, self.seed)

        # 行列への変換(結合構造のみ）
        connection = nx.to_numpy_array(G)
        W = np.array(connection)

        # 非ゼロ要素を一様分布に従う乱数として生成
        rec_scale = 1.0
        np.random.seed(seed=self.seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        # スペクトル半径の計算
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))

        # 指定のスペクトル半径rhoに合わせてスケーリング
        W *= rho / sp_radius

        return W

    # リザバー状態ベクトルの更新
    def __call__(self, x_in):
        '''
        param x_in: 更新前の状態ベクトル
        return: 更新後の状態ベクトル
        '''
        self.x = (1.0 - self.alpha) * self.x + self.alpha * torch.tanh(torch.mv(self.W, self.x) + x_in)
        return self.x

    # リザバー状態ベクトルの初期化
    def reset_reservoir_state(self):
        self.x *= 0.0


# 出力層
class Output:
    # 出力結合重み行列の初期化
    def __init__(self, N_x, N_y, seed=0):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param seed: 乱数の種
        '''
        # 正規分布に従う乱数
        np.random.seed(seed=seed)
        self.Wout = np.random.normal(size=(N_y, N_x)).astype(np.float32)
        self.Wout = torch.from_numpy(self.Wout).to(device)

    # 出力結合重み行列による重みづけ
    def __call__(self, x):
        '''
        param x: N_x次元のベクトル
        return: N_y次元のベクトル
        '''
        return torch.mv(self.Wout, x)

    # 学習済みの出力結合重み行列を設定
    def setweight(self, Wout_opt):
        self.Wout = Wout_opt


# リッジ回帰（beta=0のときは線形回帰）
class Tikhonov:
    def __init__(self, N_x, N_y, beta):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param beta: 正則化パラメータ
        '''
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
        '''
        param N_u: 入力次元
        param N_y: 出力次元
        param N_x: リザバーのノード数
        param density: リザバーのネットワーク結合密度
        param input_scale: 入力スケーリング
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: リザバーノードの活性化関数
        param fb_scale: フィードバックスケーリング（default: None）
        param fb_seed: フィードバック結合重み行列生成に使う乱数の種
        param leaking_rate: leaky integratorモデルのリーク率
        param classification: 分類問題の場合はTrue（default: False）
        param average_window: 分類問題で出力平均する窓幅（default: None）
        '''
        self.input = Input(N_u, N_x, input_scale)
        self.reservoir = Reservoir(N_x, density, rho, activation_func,
                                   leaking_rate)
        self.output = Output(N_x, N_y)
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x

    # バッチ学習
    def train(self, U, D, optimizer, trans_len = 0):
        '''
        U: 教師データの入力, データ長×N_u
        D: 教師データの出力, データ長×N_y
        optimizer: 学習器
        trans_len: 過渡期の長さ
        return: 学習前のモデル出力
        '''
        # U = U.to('cpu').detach().numpy().copy()
        # D = D.to('cpu').detach().numpy().copy()


        train_len = len(U)
        Y = []

        # 時間発展
        for n in range(train_len):
            x_in = self.input(U[n])

            # リザバー状態ベクトル
            x = self.reservoir(x_in)

            # 目標値
            d = D[n]            

            # 学習器
            if n > trans_len:  # 過渡期を過ぎたら
                optimizer(d, x)

            # 学習前のモデル出力
            y = self.output(x)
            Y.append(y)

        # 学習済みの出力結合重み行列を設定
        self.output.setweight(optimizer.get_Wout_opt())

        # モデル出力（学習前）
        return torch.cat(Y)

    # バッチ学習後の予測
    def predict(self, U):
        # U = U.to('cpu').detach().numpy().copy()
        
        test_len = len(U)
        Y_pred = []

        # 時間発展
        for n in range(test_len):
            x_in = self.input(U[n])

            # リザバー状態ベクトル
            x = self.reservoir(x_in)

            # 学習後のモデル出力
            y_pred = self.output(x)
            Y_pred.append(y_pred)

        # モデル出力（学習後）
        return torch.cat(Y_pred)

import matplotlib.pyplot as plt

def main():
    data = np.sin(np.arange(1000)/10)
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
    
    plt.plot(D_test)
    plt.plot(y.to('cpu').detach().numpy().copy())
    plt.show()
    
    
    

if __name__ == '__main__':
    main()

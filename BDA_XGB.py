#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.neighbors
import time
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import ipdb

import bda_utils


# In[2]:

def main(rs, det):
    bda_utils.setup_seed(10)


    # # 1. BDA Part
    # ## 1.a. Define BDA methodology

    # In[3]:


    def kernel(ker, X1, X2, gamma):
        K = None
        if not ker or ker == 'primal':
            K = X1
        elif ker == 'linear':
            if X2 is not None:
                K = sklearn.metrics.pairwise.linear_kernel(
                    np.asarray(X1).T, np.asarray(X2).T)
            else:
                K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
        elif ker == 'rbf':
            if X2 is not None:
                K = sklearn.metrics.pairwise.rbf_kernel(
                    np.asarray(X1).T, np.asarray(X2).T, gamma)
            else:
                K = sklearn.metrics.pairwise.rbf_kernel(
                    np.asarray(X1).T, None, gamma)
        return K


    def proxy_a_distance(source_X, target_X):
        """
        Compute the Proxy-A-Distance of a source/target representation
        """
        nb_source = np.shape(source_X)[0]
        nb_target = np.shape(target_X)[0]

        train_X = np.vstack((source_X, target_X))
        train_Y = np.hstack((np.zeros(nb_source, dtype=int),
                            np.ones(nb_target, dtype=int)))

        clf = svm.LinearSVC(random_state=0)
        clf.fit(train_X, train_Y)
        y_pred = clf.predict(train_X)
        error = metrics.mean_absolute_error(train_Y, y_pred)
        dist = 2 * (1 - 2 * error)
        return dist


    def estimate_mu(_X1, _Y1, _X2, _Y2):
        adist_m = proxy_a_distance(_X1, _X2)
        C = len(np.unique(_Y1))
        epsilon = 1e-3
        list_adist_c = []
        for i in range(1, C + 1):
            ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
            Xsi = _X1[ind_i[0], :]
            Xtj = _X2[ind_j[0], :]
            adist_i = proxy_a_distance(Xsi, Xtj)
            list_adist_c.append(adist_i)
        adist_c = sum(list_adist_c) / C
        mu = adist_c / (adist_c + adist_m)
        if mu > 1:
            mu = 1
        if mu < epsilon:
            mu = 0
        return mu


    # In[4]:


    class BDA:
        def __init__(self, kernel_type='primal', dim=30, lamb=1, mu=0.5, gamma=1, T=10, mode='BDA', estimate_mu=False):
            '''
            Init func
            :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
            :param dim: dimension after transfer
            :param lamb: lambda value in equation
            :param mu: mu. Default is -1, if not specificied, it calculates using A-distance
            :param gamma: kernel bandwidth for rbf kernel
            :param T: iteration number
            :param mode: 'BDA' | 'WBDA'
            :param estimate_mu: True | False, if you want to automatically estimate mu instead of manally set it
            '''
            self.kernel_type = kernel_type
            self.dim = dim
            self.lamb = lamb
            self.mu = mu
            self.gamma = gamma
            self.T = T
            self.mode = mode
            self.estimate_mu = estimate_mu

        def fit(self, Xs, Ys, Xt, Yt):
            '''
            Transform and Predict using 1NN as JDA paper did
            :param Xs: ns * n_feature, source feature
            :param Ys: ns * 1, source label
            :param Xt: nt * n_feature, target feature
            :param Yt: nt * 1, target label
            :return: acc, y_pred, list_acc
            '''
    #         ipdb.set_trace()
            list_acc = []
            X = np.hstack((Xs.T, Xt.T))  # X.shape: [n_feature, ns+nt]
            X_mean = np.linalg.norm(X, axis=0)  # why it's axis=0? the average of features
            X_mean[X_mean==0] = 1
            X /= X_mean
            m, n = X.shape
            ns, nt = len(Xs), len(Xt)
            e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
            C = np.unique(Ys)
            H = np.eye(n) - 1 / n * np.ones((n, n))
            mu = self.mu
            M = 0
            Y_tar_pseudo = None
            Xs_new = None
            for t in range(self.T):
                # print('\tStarting iter %i'%t)
                N = 0
                M0 = e * e.T * len(C)
    #             ipdb.set_trace()
                if Y_tar_pseudo is not None:
                    for i in range(len(C)):
                        e = np.zeros((n, 1))
                        
                        Ns = len(Ys[np.where(Ys == C[i])])
                        Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo == C[i])])

                        if self.mode == 'WBDA':
                            Ps = Ns / len(Ys)
                            Pt = Nt / len(Y_tar_pseudo)
                            alpha = Pt / Ps
    #                         mu = 1
                        else:
                            alpha = 1
                        
                        tt = Ys == C[i]
                        e[np.where(tt == True)] = 1 / Ns
    #                     ipdb.set_trace()
                        yy = Y_tar_pseudo == C[i]
                        ind = np.where(yy == True)
                        inds = [item + ns for item in ind]
                        try:
                            e[tuple(inds)] = -alpha / Nt
                            e[np.isinf(e)] = 0
                        except:
                            e[tuple(inds)] = 0  # ？
                        N = N + np.dot(e, e.T)

    #             ipdb.set_trace()
                # In BDA, mu can be set or automatically estimated using A-distance
                # In WBDA, we find that setting mu=1 is enough
                if self.estimate_mu and self.mode == 'BDA':
                    if Xs_new is not None:
                        mu = estimate_mu(Xs_new, Ys, Xt_new, Y_tar_pseudo)
                    else:
                        mu = 0
    #             ipdb.set_trace()
                M = (1 - mu) * M0 + mu * N
                M /= np.linalg.norm(M, 'fro')
    #             ipdb.set_trace()
                K = kernel(self.kernel_type, X, None, gamma=self.gamma)
                n_eye = m if self.kernel_type == 'primal' else n
                a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
                w, V = scipy.linalg.eig(a, b)
                ind = np.argsort(w)
                A = V[:, ind[:self.dim]]
                Z = np.dot(A.T, K)
                Z_mean = np.linalg.norm(Z, axis=0)  # why it's axis=0?
                Z_mean[Z_mean==0] = 1
                Z /= Z_mean
                Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
                
                global device
                model = sklearn.svm.SVC(kernel='linear').fit(Xs_new, Ys.ravel())
                Y_tar_pseudo = model.predict(Xt_new)
    #             ipdb.set_trace()
                acc = sklearn.metrics.mean_squared_error(Y_tar_pseudo, Yt)  # Yt is already in classes
                # print(acc)


            return Xs_new, Xt_new, A  #, acc, Y_tar_pseudo, list_acc


    # ## 1.b. Load Data

    # In[5]:


    Xs, Xt = bda_utils.load_data(if_weekday=1, if_interdet=1)
    Xs = Xs[:,det:det+1]
    Xt = Xt[:,det:det+1]
    Xs, Xs_min, Xs_max = bda_utils.normalize2D(Xs)
    Xt, Xt_min, Xt_max = bda_utils.normalize2D(Xt)


    # In[6]:


    # for i in range(Xs.shape[1]):
    #     plt.figure(figsize=[20,4])
    #     plt.plot(Xs[:, i])
    #     plt.plot(Xt[:, i])


    # ## 1.d. Hyperparameters

    # In[7]:


    label_seq_len = 7
    # batch_size = full batch
    seq_len = 12
    reduced_dim = 4
    inp_dim = min(Xs.shape[1], Xt.shape[1])
    label_dim = min(Xs.shape[1], Xt.shape[1])
    hid_dim = 12
    layers = 1
    lamb = 2
    MU = 0.7
    bda_dim = label_seq_len-4
    kernel_type = 'linear'

    hyper = {
        'inp_dim':inp_dim,
        'label_dim':label_dim,
        'label_seq_len':label_seq_len,
        'seq_len':seq_len,
        'reduced_dim':reduced_dim,
        'hid_dim':hid_dim,
        'layers':layers,
        'lamb':lamb,
        'MU': MU,
        'bda_dim':bda_dim,
        'kernel_type':kernel_type}

    hyper = pd.DataFrame(hyper, index=['Values'])


    # In[8]:


    hyper


    # ## 1.e. Apply BDA and get $Xs_{new}$, $Xt_{new}$ 

    # In[9]:


    Xs = Xs[:96, :]


    # In[10]:


    # [sample size, seq_len, inp_dim (dets)], [sample size, label_seq_len, inp_dim (dets)]
    Xs_3d, Ys_3d = bda_utils.sliding_window(Xs, Xs, seq_len, label_seq_len)  
    Xt_3d, Yt_3d = bda_utils.sliding_window(Xt, Xt, seq_len, label_seq_len)
    Ys_3d = Ys_3d[:, label_seq_len-1:, :]
    Yt_3d = Yt_3d[:, label_seq_len-1:, :]
    # print(Xs_3d.shape)
    # print(Ys_3d.shape)
    # print(Xt_3d.shape)
    # print(Yt_3d.shape)


    # In[11]:


    t_s = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xs_train_3d = []
    Ys_train_3d = []
    Xt_valid_3d = []
    Xt_train_3d = []
    Yt_valid_3d = []
    Yt_train_3d = []

    for i in range(Xs_3d.shape[2]):
        # print('Starting det %i'%i)
        bda = BDA(kernel_type='linear', dim=seq_len-reduced_dim, lamb=lamb, mu=MU, gamma=1, T=2)  # T is iteration time
        Xs_new, Xt_new, A = bda.fit(
            Xs_3d[:, :, i], bda_utils.get_class(Ys_3d[:, :, i]), Xt_3d[:, :, i], bda_utils.get_class(Yt_3d[:, :, i])
        )  # input shape: ns, n_feature | ns, n_label_feature
        
        # normalize
        Xs_new, Xs_new_min, Xs_new_max = bda_utils.normalize2D(Xs_new)
        Xt_new, Xt_new_min, Xt_new_max = bda_utils.normalize2D(Xt_new)
        
        # print(Xs_new.shape)
        # print(Xt_new.shape)

        day_train_t = 1
        Xs_train = Xs_new.copy()
        Ys_train = Ys_3d[:, :, i]
        Xt_valid = Xt_new.copy()[int(96*day_train_t):, :]
        Xt_train = Xt_new.copy()[:int(96*day_train_t), :]
        Yt_valid = Yt_3d[:, :, i].copy()[int(96*day_train_t):, :]
        Yt_train = Yt_3d[:, :, i].copy()[:int(96*day_train_t), :]
        

    # print('Time spent:%.5f'%(time.time()-t_s))


    # In[12]:


    # print(Xs_train.shape)
    # print(Ys_train.shape)
    # print(Xt_valid.shape)
    # print(Xt_train.shape)
    # print(Yt_valid.shape)
    # print(Yt_train.shape)


    # In[13]:


    train_x = np.vstack([Xs_train, Xt_train])
    train_y = np.vstack([Ys_train, Yt_train])


    # # 2. Regression Part

    # In[14]:


    import xgboost as xgb


    # In[15]:


    regr = xgb.XGBRegressor(base_score=0.5, booster='gbtree', 
                        colsample_bylevel=1, colsample_bynode=1, 
                        colsample_bytree=1, gamma=0, importance_type='gain', 
                        learning_rate=0.1, max_delta_step=0, max_depth=3, 
                        min_child_weight=1, missing=1, n_estimators=100, 
                        n_jobs=1, nthread=None, objective='reg:squarederror', 
                        random_state=1, reg_alpha=0, reg_lambda=1, 
                        scale_pos_weight=1, seed=None, silent=None, 
                        subsample=1, verbosity=1, tree_method="hist"
                       )

    regr.fit(train_x, train_y.flatten())


    # # 3. Evaluation

    # In[16]:


    g_t = Yt_valid.flatten()
    pred = regr.predict(Xt_valid)

    # plt.figure(figsize=[16,4])
    # plt.plot(g_t, label='label')
    # plt.plot(pred, label='predict')
    # plt.legend()


    # In[17]:


    nrmse = bda_utils.nrmse_loss_func(pred, g_t, 0)
    mape = bda_utils.mape_loss_func(pred, g_t, 0)
    smape = bda_utils.smape_loss_func(pred, g_t, 0)
    mae = bda_utils.mae_loss_func(pred, g_t, 0)
    nmae = bda_utils.nmae_loss_func(pred, g_t, 0)

    return nrmse, mape, smape, mae, nmae

mean_data = []
std_data = []
for det in range(10):
    data_det = []
    for rs in range(20):
        print(det, rs)
        nrmse, mape, smape, mae, nmae = main(rs, det)
        data_det.append([nrmse, mape, smape, mae, nmae])

    data_det = pd.DataFrame(data_det)
    mean_data.append(list(data_det.mean().values))
    std_data.append(list(data_det.std().values))

pd.DataFrame(mean_data, columns=['nrmse', 'mape', 'smape', 'mae', 'nmae']).to_csv('./table_data/bda_XGB_mean.csv')
pd.DataFrame(std_data, columns=['nrmse', 'mape', 'smape', 'mae', 'nmae']).to_csv('./table_data/bda_XGB_std.csv')






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

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import ipdb

import bda_utils


# In[2]:


bda_utils.setup_seed(10)


# # 1. BDA Part
# ## 1.a. Define BDA methodology

# In[3]:

def main(seq_len, reduced_dim):
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
                print('\tStarting iter %i'%t)
                N = 0
                M0 = e * e.T * len(C)
    #             ipdb.set_trace()
                if Y_tar_pseudo is not None:
                    for i in range(len(C)):
                        e = np.zeros((n, 1))
                        
                        Ns = len(Ys[np.where(Ys == C[i])])
                        Nt = len(Y_tar_pseudo[np.where(Y_tar_pseudo == C[i])])
    #                     Ns = 1
    #                     Nt = 1

                        alpha = 1  # bda
                        
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
                model = sklearn.neighbors.KNeighborsClassifier().fit(Xs_new, Ys.ravel())
                Y_tar_pseudo = model.predict(Xt_new)
    #             ipdb.set_trace()
                acc = sklearn.metrics.accuracy_score(Y_tar_pseudo, Yt)  # Yt is already in classes
                # print(acc)


            return Xs_new, Xt_new, A  #, acc, Y_tar_pseudo, list_acc


    # ## 1.b. Load Data

    # In[46]:


    Xs, Xt = bda_utils.load_data(if_weekday=1, if_interdet=1)
    Xs = Xs[:, :1]
    Xt = Xt[:, :1]
    Xs, Xs_min, Xs_max = bda_utils.normalize2D(Xs)
    Xt, Xt_min, Xt_max = bda_utils.normalize2D(Xt)




    # ## 1.d. Hyperparameters

    # In[86]:


    label_seq_len = 3
    # batch_size = full batch
    seq_len = 48
    reduced_dim = 15
    inp_dim = min(Xs.shape[1], Xt.shape[1])
    label_dim = min(Xs.shape[1], Xt.shape[1])
    hid_dim = 12
    layers = 1
    lamb = 3

    hyper = {
        'inp_dim':inp_dim,
        'label_dim':label_dim,
        'label_seq_len':label_seq_len,
        'seq_len':seq_len,
        'reduced_dim':reduced_dim,
        'hid_dim':hid_dim,
        'layers':layers,
        'lamb':lamb}
    hyper = pd.DataFrame(hyper, index=['Values'])


    # In[87]:


    hyper


    # ## 1.e. Apply BDA and get $Xs_{new}$, $Xt_{new}$ 

    # In[88]:


    # [sample size, seq_len, inp_dim (dets)], [sample size, label_seq_len, inp_dim (dets)]
    Xs_3d, Ys_3d = bda_utils.sliding_window(Xs, Xs, seq_len, 1)  
    Xt_3d, Yt_3d = bda_utils.sliding_window(Xt, Xt, seq_len, 1)


    # In[89]:


    t_s = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xs_train_3d = []
    Ys_train_3d = []
    Xt_valid_3d = []
    Xt_train_3d = []
    Yt_valid_3d = []
    Yt_train_3d = []

    for i in range(Xs_3d.shape[2]):
        bda = BDA(kernel_type='linear', dim=seq_len-reduced_dim, lamb=lamb, mu=0.6, gamma=1, T=1)  # T is iteration time
        Xs_new, Xt_new, A = bda.fit(
            Xs_3d[:, :, i], bda_utils.get_class(Ys_3d[:, :, i]), Xt_3d[:, :, i], bda_utils.get_class(Yt_3d[:, :, i])
        )  # input shape: ns, n_feature | ns, n_label_feature


        day_train_t = 1
        Xs_train = Xs_new.copy()
        Ys_train = Ys_3d[:, :, i]
        Xt_valid = Xt_new.copy()[int(96*day_train_t):, :]
        Xt_train = Xt_new.copy()[:int(96*day_train_t), :]
        Yt_valid = Yt_3d[:, :, i].copy()[int(96*day_train_t):, :]
        Yt_train = Yt_3d[:, :, i].copy()[:int(96*day_train_t), :]
        
        Xs_train_3d.append(Xs_train)
        Ys_train_3d.append(Ys_train)
        Xt_valid_3d.append(Xt_valid)
        Xt_train_3d.append(Xt_train)
        Yt_valid_3d.append(Yt_valid)
        Yt_train_3d.append(Yt_train)


    Xs_train_3d = np.array(Xs_train_3d)
    Ys_train_3d = np.array(Ys_train_3d)
    Xt_valid_3d = np.array(Xt_valid_3d)
    Xt_train_3d = np.array(Xt_train_3d)
    Yt_valid_3d = np.array(Yt_valid_3d)
    Yt_train_3d = np.array(Yt_train_3d)

    # In[90]:


    Xs_train_3d = np.transpose(Xs_train_3d, (1, 2, 0))
    Ys_train_3d = np.transpose(Ys_train_3d, (1, 2, 0))
    Xt_valid_3d = np.transpose(Xt_valid_3d, (1, 2, 0))
    Xt_train_3d = np.transpose(Xt_train_3d, (1, 2, 0))
    Yt_valid_3d = np.transpose(Yt_valid_3d, (1, 2, 0))
    Yt_train_3d = np.transpose(Yt_train_3d, (1, 2, 0))


    Xs_train_3d.shape

    class traff_net(nn.Module):
        def __init__(self, seq_len, hid_dim=12, layers=3):
            super(traff_net, self).__init__()

            self.seq_len = seq_len
            
            self.fc = nn.Sequential(
                nn.Linear(seq_len, seq_len*8),
                nn.ReLU(),
                nn.Linear(seq_len*8, seq_len*32),
                nn.ReLU(),
                nn.Linear(seq_len*32, seq_len*64),
                nn.ReLU(),
                nn.Linear(seq_len*64, 100+1),  # 101 classes (0-101)
                nn.ReLU(),
            )  # regression
        
        def forward(self, x):
            
            y = nn.Flatten()(x)
            y = self.fc(y)  # fully connected layer
            return y


    batch_size = 1960

    train_x = np.vstack([Xs_train_3d, Xt_train_3d])
    train_y = np.vstack([Ys_train_3d, Yt_train_3d])

    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    Xt_valid_3d = torch.tensor(Xt_valid_3d, dtype=torch.float32).to(device)
    Yt_valid_3d = torch.tensor(Yt_valid_3d, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)
    train_iter = iter(train_loader)

    # build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = traff_net(seq_len - reduced_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    #scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, 0.7)
    train_loss_set = []
    val_loss_set = []

    det = 0  # which detector to visualize


    # In[96]:


    # train
    net.train()

    epochs = 201

    for e in range(epochs):
        for i in range(len(train_loader)):
            try:
                data, label = train_iter.next()
            except:
                train_iter = iter(train_loader)
                data, label = train_iter.next()

            out = net(data)
            loss = criterion(out, bda_utils.get_class(label[:, 0, 0]).flatten().long() )  # label.shape=[batch, 1, num_dets]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            val_out = net(Xt_valid_3d)
            val_loss = criterion(val_out, bda_utils.get_class(Yt_valid_3d[:, 0, 0]).flatten().long() )
            
            val_loss_set.append(val_loss.cpu().detach().numpy())
            train_loss_set.append(loss.cpu().detach().numpy())
            
    return loss.cpu().detach().numpy(), val_loss.cpu().detach().numpy()


if __name__ == '__main__':
    l_df = pd.DataFrame(np.zeros([50, 50]))
    val_l_df = pd.DataFrame(np.zeros([50, 50]))
    for seq in range(12, 50):
        for red in range(0, seq):
            l, v_l = main(seq, red)
            l_df.iloc[seq, red] = l
            val_l_df.iloc[seq, red] = v_l
            print('\n')
            print(seq, red)
            print(l)
            print(v_l)
            print('\n')

    l_df.to_csv('l_csv')
    val_l_df.to_csv('val_l.csv')
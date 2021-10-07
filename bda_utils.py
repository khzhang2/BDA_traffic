import numpy as np
import pandas as pd
import os
import random

import torch
from torch import nn
import torch.nn.functional as F


class traff_net_reg(nn.Module):
    def __init__(self, seq_len, hid_dim=12, layers=3):
        super(traff_net_reg, self).__init__()

        self.seq_len = seq_len
        
        self.fc = nn.Sequential(
            nn.Linear(seq_len, seq_len*32),
            nn.ReLU(),
            nn.Linear(seq_len*32, 25+1),  # 101 classes (0-101)
            nn.ReLU(),
        )  # regression
    
    def forward(self, x):
        # input: (batchsize, seq_len, input_dim)
        # output: (batchsize, seq_len, hid_dim)
#         ipdb.set_trace()
        
        y = nn.Flatten()(x)
        y = self.fc(y)  # fully connected layer
#         y = F.log_softmax(y, dim=1)
        y = nn.ReLU()(y+nn.Flatten()(x))
        return y


class traff_net_clf(nn.Module):
    def __init__(self, seq_len, hid_dim=12, layers=3):
        super(traff_net_clf, self).__init__()

        self.seq_len = seq_len
        
        self.fc = nn.Sequential(
            nn.Linear(seq_len, seq_len*32),
            nn.ReLU(),
            nn.Linear(seq_len*32, seq_len*64),
            nn.ReLU(),
            nn.Linear(seq_len*64, seq_len*32),
            nn.ReLU(),
            nn.Linear(seq_len*32, seq_len*64),
            nn.ReLU(),
            nn.Linear(seq_len*64, seq_len*32),
            nn.ReLU(),
            nn.Linear(seq_len*32, 25+1),  # 101 classes (0-101)
            nn.ReLU(),
        )  # regression
    
    def forward(self, x):
        # input: (batchsize, seq_len, input_dim)
        # output: (batchsize, seq_len, hid_dim)
#         ipdb.set_trace()
        
        y = nn.Flatten()(x)
        y = self.fc(y)  # fully connected layer
#         y = F.log_softmax(y, dim=1)
        # y = nn.ReLU()(y+nn.Flatten()(x))
        return y


def mape_loss_func(preds, labels, m):
    mask = labels > m
    return np.mean(np.fabs(labels[mask]-preds[mask])/labels[mask])

def smape_loss_func(preds, labels, m):
    mask= labels > m
    return np.mean(2*np.fabs(labels[mask]-preds[mask])/(np.fabs(labels[mask])+np.fabs(preds[mask])))

def mae_loss_func(preds, labels, m):
    mask= labels > m
    return np.mean(np.fabs((labels[mask]-preds[mask])))

def nrmse_loss_func(preds, labels, m):
    mask= labels > m
    return np.sqrt(np.sum((preds[mask] - labels[mask])**2)/preds[mask].flatten().shape[0])/(labels[mask].max() - labels[mask].min())

def eliminate_nan(b):
    a = np.array(b)
    c = a[~np.isnan(a)]
    return c


def get_class(v):
    # v is 1-d or 2-d array
    # we set that there are 100 classes between 0 and 1
    if len(v.shape) == 1:
        try:
            v = v.reshape(-1, v.shape[0])
        except:
            v = v.view(-1, v.shape[0])

    try:
        v = np.array(v)
        v_cls = np.zeros_like(v)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_cls[i, j] = int(np.floor(v[i, j]*100))//4
        return v_cls
    except:
    #     None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        v_cls = torch.zeros_like(v)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v_cls[i, j] = int(torch.floor(v[i, j]*100))//4
        return v_cls


def normalize2D(V):
    V = np.array(V)
    return ( V - V.min(0) ) / ( V.max(0) - V.min(0) ), V.min(0), V.max(0)


def denormalize2D(V, V_min, V_max):
    V = np.array(V)
    V_min = np.array(V_min)
    V_max = np.array(V_max)
    denormalized_V = V * (V_max - V_min) + V_min
    return denormalized_V


def sliding_window(T, T_org, seq_len, label_seq_len):  # was: (T, T_org, seq_len, label_seq_len)
    # seq_len is equal to window_size
    # T (np.ndarray) has dim: sample size, dim
    K = T.shape[0] - seq_len - label_seq_len + 1  # Li, et al., 2021, TRJ part C, pp. 8
    
#     TT_org = T_org.reshape(-1, 1)

    # assemble the data into 3D
    x_set = T[:K, np.newaxis, :]
#     x_set = np.concatenate(TT[i : K+i, 0] for i in range(seq_len), axis=1)
    for i in range(1, seq_len):
        x_set = np.concatenate((x_set, T[i:K+i, np.newaxis, :]), axis=1)
    
    y_set = T_org[seq_len:K+seq_len, np.newaxis, :]
    for i in range(1, label_seq_len):
        y_set = np.concatenate((y_set, T_org[i+seq_len:K+i+seq_len, np.newaxis, :]), axis=1)
    
#     y_set = np.vstack(T_org[i+seq_len : K+seq_len+i, 0] for i in range(label_seq_len)).T
    
    assert x_set.shape[0] == y_set.shape[0]

    # return size: n_samp, seq_len
    return x_set, y_set


def load_data(if_weekday=1, if_interdet=1):
    file_set_2020 = [files for root, dirs, files in os.walk('./data/')][0][1::2]
    # file_set_2020 = [files for root, dirs, files in os.walk('./data/')][0][2::2]

    from_date = 55
    to_date = 67

    det_num = 23+13 if if_interdet==1 else 23
    src_det_ind = 23 if if_interdet==1 else 13

    if if_interdet==1:
        src_data = np.zeros([to_date - from_date, 96, 23])  # num_days, time_seg_per_day, num_dets
        tar_data = np.zeros([to_date - from_date, 96, 13])  # num_days, time_seg_per_day, num_dets
    else:
        src_data = np.zeros([to_date - from_date, 96, 13])  # num_days, time_seg_per_day, num_dets
        tar_data = np.zeros([to_date - from_date, 96, 10])  # num_days, time_seg_per_day, num_dets

    # choosing date, 
    weekdays = np.array([6,7,8,9,10])
    weekends = np.array([4,5,11])
    day_type = weekdays if if_weekday else weekends
    src_data = src_data[day_type, :, :]
    tar_data = tar_data[day_type, :, :]

    for i in range(det_num):  # 23 M1 csv
        # from_date 和 to_date
        if i<src_det_ind:
            src_data[:, :, i] += np.array(pd.read_csv('./data/'+file_set_2020[i]).iloc[from_date:to_date, 1:-1])[day_type, :]
        else:
            tar_data[:, :, i-src_det_ind] += np.array(pd.read_csv('./data/'+file_set_2020[i]).iloc[from_date:to_date, 1:-1])[day_type, :]

    if if_interdet:
        Xs = src_data.reshape(-1, 23)[:, :13][:, np.array([0,1,2,3,5,6,8,9,10,11])]  # choosing detectors
        Xt = tar_data.reshape(-1, 13)[:, np.array([0,1,2,3,5,6,8,9,10,11])]
    else:
        Xs = src_data.reshape(-1, 13)[:, np.array([0,1,2,3,5,6,8,9,10,11])]
        Xt = tar_data.reshape(-1, 10)

    # Xs = (Xs - Xs.min(0))/(Xs.max(0)-Xs.min(0))
    # Xt = (Xt - Xt.min(0))/(Xt.max(0)-Xt.min(0))
    return Xs, Xt


def setup_seed(seed):
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)    
        torch.backends.cudnn.deterministic = True
    except:
        tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_np(array, name):
    np.savetxt(name, array, delimiter=',')


def get_num():
    return len(next(iter(os.walk('./outputs/BDA/')))[2])
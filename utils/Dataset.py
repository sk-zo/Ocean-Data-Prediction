import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler

class SequentialDataset(data.Dataset):
    def __init__(self, data_path, seq_len, feature_len=11, scale=False, mode='train'):
        self.inputs = []
        self.labels = []
        self.mode = mode

        df = pd.read_csv(data_path, encoding='euc-kr')
        df = df.iloc[:, 4:]
        if feature_len == 11:
            df.drop(['전기전도도 (mS/㎝)', '총질소(mg/L)', '총인(mg/L)', '클로로필(㎍/L)', '일사량(W/㎡)', '풍향', '강수량(mm)'], axis=1, inplace=True)
        elif feature_len == 10:
            df.drop(['전기전도도 (mS/㎝)', '풍속(m/sec)', '총질소(mg/L)', '총인(mg/L)', '클로로필(㎍/L)', '일사량(W/㎡)', '풍향', '강수량(mm)'], axis=1, inplace=True)
        elif feature_len == 9:
            df.drop(['전기전도도 (mS/㎝)', '탁도(NTU)', '풍속(m/sec)', '총질소(mg/L)', '총인(mg/L)', '클로로필(㎍/L)', '일사량(W/㎡)', '풍향', '강수량(mm)'], axis=1, inplace=True)            
        elif feature_len == 8:
            df.drop(['전기전도도 (mS/㎝)', '탁도(NTU)', '풍속(m/sec)', '총질소(mg/L)', '총인(mg/L)', '클로로필(㎍/L)', '일사량(W/㎡)', '풍향', '강수량(mm)', '화학적산소요구량(mg/L)'], axis=1, inplace=True)            
        elif feature_len == 7:
            df.drop(['전기전도도 (mS/㎝)', '탁도(NTU)', '풍속(m/sec)', '총질소(mg/L)', '총인(mg/L)', '클로로필(㎍/L)', '일사량(W/㎡)', '풍향', '강수량(mm)', '화학적산소요구량(mg/L)', '질산질소(mg/L)'], axis=1, inplace=True)            
        df.bfill(inplace=True)
        if scale:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(df).tolist()
        else: data = df.values.tolist()
        
        if mode == 'train':
            for i in tqdm(range(len(data)-seq_len)):
                self.inputs.append(data[i:i+seq_len])
                self.labels.append(data[i+seq_len])

        self.len = len(self.inputs)

    def __getitem__(self, index):
        return np.array(self.inputs[index]), np.array(self.labels[index])

    def __len__(self):
        return self.len


class SequentialDatasetH(data.Dataset):
    def __init__(self, data_path, seq_len, feature_len=11, scale=False, mode='train'):
        self.inputs = []
        self.labels = []
        self.mode = mode

        df = pd.read_csv(data_path, encoding='euc-kr')
        df = df.iloc[:, 5:]
        if feature_len == 11:
            df.drop(['전기전도도 (mS/㎝)', '총질소(mg/L)', '총인(mg/L)', '클로로필(㎍/L)', '일사량(W/㎡)', '풍향', '강수량(mm)'], axis=1, inplace=True)
        elif feature_len == 10:
            df.drop(['전기전도도 (mS/㎝)', '풍속(m/sec)', '총질소(mg/L)', '총인(mg/L)', '클로로필(㎍/L)', '일사량(W/㎡)', '풍향', '강수량(mm)'], axis=1, inplace=True)
        elif feature_len == 9:
            df.drop(['전기전도도 (mS/㎝)', '탁도(NTU)', '풍속(m/sec)', '총질소(mg/L)', '총인(mg/L)', '클로로필(㎍/L)', '일사량(W/㎡)', '풍향', '강수량(mm)'], axis=1, inplace=True)            
        elif feature_len == 8:
            df.drop(['전기전도도 (mS/㎝)', '탁도(NTU)', '풍속(m/sec)', '총질소(mg/L)', '총인(mg/L)', '클로로필(㎍/L)', '일사량(W/㎡)', '풍향', '강수량(mm)', '화학적산소요구량(mg/L)'], axis=1, inplace=True)            
        df.bfill(inplace=True)
        if scale:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(df).tolist()
        else: data = df.values.tolist()
        
        if mode == 'train':
            for i in tqdm(range(len(data)-seq_len)):
                self.inputs.append(data[i:i+seq_len])
                self.labels.append(data[i+seq_len])

        self.len = len(self.inputs)

    def __getitem__(self, index):
        return np.array(self.inputs[index]), np.array(self.labels[index])

    def __len__(self):
        return self.len
        

class SequentialDataset20(data.Dataset):
    def __init__(self, data_path, seq_len, feature_len=11, scale=False, mode='train'):
        self.inputs = []
        self.labels = []
        self.mode = mode

        df = pd.read_csv(data_path, encoding='euc-kr')
        df = df.iloc[:, 1:]
        if feature_len == 11:
            df.drop(['염분계 - 전기전도도', '멀티센서 - 남조류', 'CNP - 총질소', 'CNP - 총인', '멀티센서 - 클로로필', '기상센서 - 일사량', '기상센서 - 풍향', '기상센서 - 강수량'], axis=1, inplace=True)
        elif feature_len == 10:
            df.drop(['염분계 - 전기전도도', '기상센서 - 풍속', '멀티센서 - 남조류', 'CNP - 총질소', 'CNP - 총인', '멀티센서 - 클로로필', '기상센서 - 일사량', '기상센서 - 풍향', '기상센서 - 강수량'], axis=1, inplace=True)
        elif feature_len == 9:
            df.drop(['염분계 - 전기전도도', '멀티센서 - 탁도', '기상센서 - 풍속', '멀티센서 - 남조류', 'CNP - 총질소', 'CNP - 총인', '멀티센서 - 클로로필', '기상센서 - 일사량', '기상센서 - 풍향', '기상센서 - 강수량'], axis=1, inplace=True)
        elif feature_len == 8:
            df.drop(['염분계 - 전기전도도', '멀티센서 - 탁도', '기상센서 - 풍속', '멀티센서 - 남조류', 'CNP - 총질소', 'CNP - 총인', '멀티센서 - 클로로필', '기상센서 - 일사량', '기상센서 - 풍향', '기상센서 - 강수량', 'CNP - 화학적산소요구량'], axis=1, inplace=True)
        elif feature_len == 7:
            df.drop(['염분계 - 전기전도도', '멀티센서 - 탁도', '기상센서 - 풍속', '멀티센서 - 남조류', 'CNP - 총질소', 'CNP - 총인', '멀티센서 - 클로로필', '기상센서 - 일사량', '기상센서 - 풍향', '기상센서 - 강수량', 'CNP - 화학적산소요구량', '영양염류 - 질산질소'], axis=1, inplace=True)
        df.bfill(inplace=True)
        if scale:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(df).tolist()
        else: data = df.values.tolist()
        
        if mode == 'train':
            for i in tqdm(range(len(data)-seq_len-1)):
                self.inputs.append(data[i:i+seq_len])
                self.labels.append(data[i+seq_len])

        self.len = len(self.inputs)

    def __getitem__(self, index):
        return np.array(self.inputs[index]), np.array(self.labels[index])

    def __len__(self):
        return self.len


class SequentialDataset20H(data.Dataset):
    def __init__(self, data_path, seq_len, feature_len=11, scale=False, mode='train'):
        self.inputs = []
        self.labels = []
        self.mode = mode

        df = pd.read_csv(data_path, encoding='euc-kr')
        df = df.iloc[:, 2:]
        if feature_len == 11:
            df.drop(['염분계 - 전기전도도', '멀티센서 - 남조류', 'CNP - 총질소', 'CNP - 총인', '멀티센서 - 클로로필', '기상센서 - 일사량', '기상센서 - 풍향', '기상센서 - 강수량'], axis=1, inplace=True)
        elif feature_len == 10:
            df.drop(['염분계 - 전기전도도', '기상센서 - 풍속', '멀티센서 - 남조류', 'CNP - 총질소', 'CNP - 총인', '멀티센서 - 클로로필', '기상센서 - 일사량', '기상센서 - 풍향', '기상센서 - 강수량'], axis=1, inplace=True)
        elif feature_len == 9:
            df.drop(['염분계 - 전기전도도', '멀티센서 - 탁도', '기상센서 - 풍속', '멀티센서 - 남조류', 'CNP - 총질소', 'CNP - 총인', '멀티센서 - 클로로필', '기상센서 - 일사량', '기상센서 - 풍향', '기상센서 - 강수량'], axis=1, inplace=True)
        elif feature_len == 8:
            df.drop(['염분계 - 전기전도도', '멀티센서 - 탁도', '기상센서 - 풍속', '멀티센서 - 남조류', 'CNP - 총질소', 'CNP - 총인', '멀티센서 - 클로로필', '기상센서 - 일사량', '기상센서 - 풍향', '기상센서 - 강수량', 'CNP - 화학적산소요구량'], axis=1, inplace=True)
        df.bfill(inplace=True)
        if scale:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(df).tolist()
        else: data = df.values.tolist()
        
        if mode == 'train':
            for i in tqdm(range(len(data)-seq_len-1)):
                self.inputs.append(data[i:i+seq_len])
                self.labels.append(data[i+seq_len])

        self.len = len(self.inputs)

    def __getitem__(self, index):
        return np.array(self.inputs[index]), np.array(self.labels[index])

    def __len__(self):
        return self.len

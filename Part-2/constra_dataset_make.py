import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import multiprocessing


def get_data():
    df = pd.read_pickle('data_n.pkl')

    dodf = ()
    for i in df["action"].unique():
        dodf[i] = df.loc[df['action'] == i].sample(n=5024)

    print("Number of cpu : ", multiprocessing.cpu_count())
    return dodf

def init_data():
    dat1 = pd.DataFrame(columns = ['image1', 'language1','image2', 'language2','action'])
    return dat1

def get_dt(start, final, return_dict):
    dic = get_data()
    dat1 = init_data()
    for i in range(len(dic)):
        for j in range(len(dic[i]) - 1):
            dat1 = dat1.append({'image1': dic[i].iloc[j]['image'], 'language1': dic[i].iloc[j]['language'], 'image2': dic[i].iloc[j+1]['image'], 'language2': dic[i].iloc[j+1]['language'], 'action': 1}, ignore_index=True)
    for i in [0,2]:
        for j in [0,2]:
            if i != j:
                for k in range(start,final -1 ):
                    dat1 = dat1.append({'image1': dic[i].iloc[k]['image'], 'language1': dic[i].iloc[k]['language'], 'image2': dic[j].iloc[k+1]['image'], 'language2': dic[j].iloc[k+1]['language'], 'action': 0}, ignore_index=True)
    return dat1

if __name__ == '__main__':
    dff = get_dt(0, 1882)

    #dff = pd.concat(return_dict.values(), ignore_index=True)

    dff.to_pickle("conTra_data1.pkl")
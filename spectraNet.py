import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = pd.read_csv('data/Barley.data.csv')

X = data.iloc[:, 1:]
y = data.iloc[:, :1]

class SpectraNet(nn.Module):
    def __init__(self, input_size):
        super(SpectraNet, self).__init__()

        self.c1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8)
        self.m1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.c2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16)
        self.m2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.f1 = nn.Flatten()
        self.d1 = nn.Dropout(0.5)
        self.l1 = nn.Linear() # Need to understand how many neurons were involved
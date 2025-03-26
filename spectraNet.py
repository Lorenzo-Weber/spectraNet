import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

data = pd.read_csv('data/Barley.data.csv')

X = data.iloc[:, 1:]
y = data.iloc[:, :1]

encoder = OneHotEncoder()

y_encoded = encoder.fit_transform(y) # FIX


x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

data_pipe = Pipeline([
    # SNV, STS, Noise, 1HOT, Data Aug.?
])

class SpectraNet(nn.Module):
    def __init__(self, input_size = None):
        super(SpectraNet, self).__init__()

        self.c1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8)
        self.m1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.c2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16)
        self.m2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.f1 = nn.Flatten()
        self.d1 = nn.Dropout(0.5)
        self.l1 = nn.Linear(100,100) # Need to understand how many neurons were involved
        self.l2 = nn.Linear(100,100) # Not sure too
        self.out = nn.Linear(100,28) # Fix the output

    def forward(self, x):
        x = F.elu(self.c1(x))
        x = self.m1(x)
        x = F.elu(self.c2(x))
        x = self.m2(x)
        x = self.f1(x)
        x = self.d1(x)
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))

        x = self.out(x)

instances = X.iloc[:20]

plt.figure(figsize=(10, 10))
for i, instance in instances.iterrows():
    plt.plot(instance.index, instance.values, label=f"Waveform {i}")

plt.title("Waveforms of Multiple Instances")
plt.xlabel("Wavelength Index")
plt.ylabel("Intensity")
plt.legend()
plt.show()
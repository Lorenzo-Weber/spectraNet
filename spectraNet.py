import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import recall_score

def snv(input_data):
    mean = np.mean(input_data, axis=1, keepdims=True) 
    std = np.std(input_data, axis=1, keepdims=True)  
    return (input_data - mean) / std

def gaussian_noise(input_data, mean=0, std=0.01):
    noise = np.random.normal(mean,std, input_data.shape)
    return input_data + noise

class GauNoiseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        x_noisy =  gaussian_noise(X)
        return x_noisy

class SNVTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        x_new =  snv(X)
        return x_new

data = pd.read_csv('data/Barley.data.csv')

X = data.iloc[:, 1:]
y = data.iloc[:, :1]
X_arr = X.to_numpy()

encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y).toarray()

data_pipeline = Pipeline([
    ('snv', SNVTransformer()),
    ('scaler', StandardScaler()),
    ('noise', GauNoiseTransformer())
])

x_ready = data_pipeline.fit_transform(X_arr)

x_train, x_test, y_train, y_test = train_test_split(x_ready, y_encoded, test_size=0.2)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class SpectraNet(nn.Module):
    def __init__(self, input_size = None):
        super(SpectraNet, self).__init__()

        self.c1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8)
        self.m1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.c2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16)
        self.m2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.f1 = nn.Flatten()
        self.d1 = nn.Dropout(0.5)
        self.l1 = nn.Linear(2336, 512)
        self.l2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 24) 

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):

        x = x.unsqueeze(1)
        x = F.elu(self.c1(x))
        x = self.m1(x)
        x = F.elu(self.c2(x))
        x = self.m2(x)
        x = self.f1(x)
        x = self.d1(x)
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))

        return self.out(x)


def evaluate(model, test_loader, criterion):
    model.eval() 
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():  
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)  
            true_labels = torch.argmax(labels, dim=1)  

            correct += (predicted == true_labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    recall = 100 * recall_score(all_labels, all_preds, average='macro') 

    print(f'Teste - Loss: {avg_loss:.4f}, Acurácia: {accuracy:.2f}%, Recall: {recall:.4f}%')

    return avg_loss, accuracy, recall


model = SpectraNet()
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.0004, weight_decay=1e-5)

epochs = 45

for epoch in range(epochs):
    model.train()
    run_loss = 0.0

    for inputs, labels in train_loader:
        opt.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        run_loss += loss
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {run_loss / len(train_loader):.4f}")

test_loss, test_acc, recall = evaluate(model, test_loader, criterion)

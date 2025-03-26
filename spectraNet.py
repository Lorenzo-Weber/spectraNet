import pandas as pd

data = pd.read_csv('data/Barley.data.csv')

X = data.iloc[:, 1:]
y = data.iloc[:, :1]


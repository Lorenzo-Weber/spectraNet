import pandas as pd
import numpy as np
from sklearn.utils import resample

data = pd.read_csv('data/Barley.csv')

# Agrupar os dados por 'Predictor'
groups = data.groupby('Predictor')
predictors = {group: df for group, df in groups}

# Técnicas de Data Augmentation
def add_gaussian_noise(data):
    """Adiciona ruído gaussiano com variação aleatória no nível"""
    noise_level = np.random.uniform(0.005, 0.02)  # Aleatório entre 0.5% e 2%
    noise = np.random.normal(0, noise_level, data.shape)
    return np.clip(data + noise, a_min=0, a_max=np.max(data))

def add_uniform_noise(data):
    """Adiciona ruído de distribuição uniforme"""
    noise = np.random.uniform(-0.02, 0.02, data.shape)
    return np.clip(data + noise, a_min=0, a_max=np.max(data))

def mixup(data):
    """Cria novas amostras como combinação de duas existentes"""
    indices = np.random.choice(data.shape[0], size=(data.shape[0], 2))
    alpha = np.random.beta(0.4, 0.4, size=(data.shape[0], 1))  # Ponderação aleatória
    mixed_data = alpha * data[indices[:, 0]] + (1 - alpha) * data[indices[:, 1]]
    return np.clip(mixed_data, a_min=0, a_max=np.max(data))

def log_transform(data):
    """Aplica transformação log(1+x) para criar variações"""
    return np.log1p(data)

def bootstrap_resample(data):
    """Cria novas amostras por reamostragem com reposição"""
    return resample(data, replace=True, n_samples=data.shape[0])

# Criar novas instâncias
augmented_data = []

for predictor, group in predictors.items():
    numeric_cols = group.select_dtypes(include=[np.number]).columns
    group_data = group[numeric_cols].to_numpy()

    for _ in range(2):  # Criar 2 variações de cada técnica para cada amostra
        augmented_data.append(pd.DataFrame(add_gaussian_noise(group_data), columns=numeric_cols).assign(Predictor=predictor))
        augmented_data.append(pd.DataFrame(add_uniform_noise(group_data), columns=numeric_cols).assign(Predictor=predictor))
        augmented_data.append(pd.DataFrame(mixup(group_data), columns=numeric_cols).assign(Predictor=predictor))
        augmented_data.append(pd.DataFrame(log_transform(group_data), columns=numeric_cols).assign(Predictor=predictor))
        augmented_data.append(pd.DataFrame(bootstrap_resample(group_data), columns=numeric_cols).assign(Predictor=predictor))

augmented_data = pd.concat(augmented_data, ignore_index=True)
final_data = pd.concat([data, augmented_data], ignore_index=True)

final_data.to_csv('data/Barley_augmented.csv', index=False)

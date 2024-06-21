import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')

selected = ['f_loc_v', 'f_acc_x_mean', 'f_acc_x_std', 'f_acc_y_zero_crossings', 'w_acc_x_mean', 'w_acc_x_min',
            'w_acc_x_max', 'w_acc_y_mean', 'w_acc_y_max', 'w_acc_y_zero_crossings', 'w_acc_z_mean', 'w_acc_z_max',
            'w_acc_z_zero_crossings', 'f_gyr_x_mean', 'f_gyr_x_max', 'f_gyr_x_skew', 'f_gyr_y_mean', 'f_gyr_y_max',
            'f_gyr_z_mean', 'f_gyr_z_max', 'w_gyr_x_min', 'w_gyr_y_mean', 'w_gyr_y_min', 'f_loc_v_mean', 'f_loc_v_min',
            'f_loc_v_max', 'w_loc_h_max', 'w_loc_v_mean', 'w_loc_v_min', 'w_loc_d_mean', 'w_loc_d_std', 'w_loc_d_min',
            'w_loc_d_max', 'f_mag_x_mean', 'f_mag_x_std', 'f_mag_x_min', 'f_mag_x_max', 'f_mag_y_std', 'f_mag_y_min',
            'f_mag_y_max', 'f_mag_y_zero_crossings', 'f_mag_z_mean', 'f_mag_z_std', 'f_mag_z_min', 'f_mag_z_max',
            'w_mag_x_mean', 'w_mag_x_min', 'w_mag_x_max', 'w_mag_y_min', 'w_mag_z_mean', 'w_mag_z_std', 'w_mag_z_min',
            'w_mag_z_max', 'w_mag_z_zero_crossings', 'hr_max', 'f_acc_x_fft_mean', 'w_acc_z_fft_mean',
            'f_gyr_z_fft_mean', 'f_loc_v_fft_mean', 'w_loc_h_fft_mean', 'w_loc_d_fft_mean', 'f_mag_x_fft_mean',
            'f_mag_z_fft_mean', 'w_mag_z_fft_mean']

X = X[selected]
y = y['act']

def create_sequences(X, y, sequence_length, window_length=50):
    tot = []
    for base in range(0, len(y), window_length):
        item = min(window_length, len(y) - base)
        if item < sequence_length:
            continue
        this_s = []
        for i in range(item - sequence_length):
            seq = X[base + i:base + i + sequence_length]
            label = y[base + i + sequence_length - 1]

            this_s.append((seq, label))
        random.shuffle(this_s)
        tot.append(this_s)
    random.shuffle(tot)
    tot = [item for sublist in tot for item in sublist]
    X, y = zip(*tot)

    return np.array(X), np.array(y)


def normalize(X, y):
    X_flat = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    X_scaled_reshaped = X_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])
    return X_scaled_reshaped, y


X, y = create_sequences(X, y, 10)
X, y = normalize(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

batch_size = 16
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

################################################


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Hyperparameters
input_size = 64
hidden_size = 64
num_classes = 5
num_layers = 1

# Model
model = LSTMModel(input_size, hidden_size, num_classes, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


model.eval()
l_pre, l_act = [], []
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        l_pre.append(predicted.item())
        l_act.append(labels.item())

print('lstm_predict_4 =', l_pre)
print('lstm_actual_4 =', l_act)


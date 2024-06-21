import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import random
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

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_classes, num_channels, kernel_size=2, dropout=0.2):
        super(TCNClassifier, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        y1 = self.tcn(x)
        o = self.fc(y1[:, :, -1])
        return o

################################################

input_size = 64
num_classes = 5
num_channels = [25, 25, 25, 25]

model = TCNClassifier(input_size, num_classes, num_channels)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs.permute(0, 2, 1))
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


model.eval()
l_pre, l_act = [], []
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs.permute(0, 2, 1))
        _, predicted = torch.max(outputs.data, 1)
        l_pre.append(predicted.item())
        l_act.append(labels.item())

print('tcn_predict_0 =', l_pre)
print('tcn_actual_0 =', l_act)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

import streamlit as st
import yfinance as yf


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cuda is installed on my device

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

def load_stock_data(ticker, start_date):
    df = yf.download(ticker, start=start_date, progress=False)
    return df
    
# @st.cache_resource
def prepare_data(df, seq_length, train_ratio=0.8):
    scaler= StandardScaler()
    close_prices = df[['Close']].values
    scaled_data = scaler.fit_transform(close_prices)

    data = []
    for i in range(len(scaled_data) - seq_length):
        data.append(scaled_data[i:i+seq_length])

    data = np.array(data)
    train_size = int(train_ratio * len(data))

    # Train sets, x-input, y-output
    X_train = torch.from_numpy(data[:train_size, :-1, :]).type(torch.Tensor).to(device)
    y_train = torch.from_numpy(data[:train_size, -1, :]).type(torch.Tensor).to(device)
    # Test sets, x-input, y-outputs
    X_test = torch.from_numpy(data[train_size:, :-1, :]).type(torch.Tensor).to(device)
    y_test = torch.from_numpy(data[train_size:, -1, :]).type(torch.Tensor).to(device)

    return X_train, y_train, X_test, y_test, scaler

# @st.cache_resource
def train_model(model, X_train, y_train, num_epochs, learning_rate, callback=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    model.train()

    for i in range(num_epochs):
        y_train_pred = model(X_train)

        loss = criterion(y_train_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses

# @st.cache_resource
def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    model.eval()

    with torch.no_grad():
        y_train_pred = model(X_train)
        y_test_pred = model(X_test)

    y_train_pred_inv = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
    y_train_inv = scaler.inverse_transform(y_train.detach().cpu().numpy())
    y_test_pred_inv = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
    y_test_inv = scaler.inverse_transform(y_test.detach().cpu().numpy())
    
    # Calculate RMSE
    train_rmse = root_mean_squared_error(y_train_inv[:, 0], y_train_pred_inv[:, 0])
    test_rmse = root_mean_squared_error(y_test_inv[:, 0], y_test_pred_inv[:, 0])
    
    return {'train_predictions': y_train_pred_inv,
        'train_actual': y_train_inv,
        'test_predictions': y_test_pred_inv,
        'test_actual': y_test_inv,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse}


# @st.cache_resource
def predict_next_day(model, recent_data, scaler, seq_length):
    model.eval()
    
    # Scale data
    scaled_data = scaler.transform(recent_data[-seq_length:].reshape(-1, 1))
    
    # Convert to tensor
    x = torch.from_numpy(scaled_data[:-1]).type(torch.Tensor).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        pred = model(x)
    
    # Inverse transform
    pred_price = scaler.inverse_transform(pred.cpu().numpy())
    
    return pred_price[0, 0]
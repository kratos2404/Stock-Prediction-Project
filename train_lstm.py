import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Fetch stock data
def get_stock_data(ticker, period="2y", interval="1d"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

# Define LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last output for prediction
        return out
    

# Prepare data for training
def prepare_data(df, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    x, y = [], []
    for i in range(time_steps, len(scaled_data)):
        x.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i])
    return np.array(x), np.array(y), scaler

# Train the model
def train_model(ticker="TSLA", epochs=50, lr=0.001):
    df = get_stock_data(ticker)
    x, y, scaler = prepare_data(df)

    # Convert to tensors
    x_train = torch.tensor(x, dtype=torch.float32).to(device)
    y_train = torch.tensor(y, dtype=torch.float32).to(device)

    # Define model
    model = StockLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    # Save model
    torch.save(model.state_dict(), "lstm_stock.pth")
    print("Model saved successfully!")

    return model, scaler
if __name__ == "__main__":
    model, scaler = train_model("TSLA")  # Train model for Apple stock


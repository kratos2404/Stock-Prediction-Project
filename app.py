from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define LSTM Model (Same as training)
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
        out = self.fc(out[:, -1, :])  # Take last output for prediction
        return out

# Load trained model
model = StockLSTM().to(device)
model.load_state_dict(torch.load("lstm_stock.pth", map_location=device))
model.eval()

# Fetch stock data
def get_stock_data(ticker, period="60d", interval="1d"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

# Prepare input sequence for LSTM
def prepare_input_sequence(df, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

    input_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    input_seq = torch.tensor(input_seq, dtype=torch.float32).to(device)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
    print("Scaled Data:", scaled_data[:5])  # Check the first few values

    return input_seq, scaler
# Route to serve the frontend UI
@app.route('/')
def index():
    return render_template('index.html')

# API Route to get stock predictions
@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        data = request.get_json()
        ticker = data.get("ticker", "").upper()
        print("Received Ticker:", ticker)  # Debugging Line

        if not ticker:
            return jsonify({"error": "Ticker symbol is required"}), 400

        df = get_stock_data(ticker)
        print("Fetched Data:", df.head())  # Debugging Line

        if df.empty:
            return jsonify({"error": "Invalid stock ticker or no data available"}), 400

        input_seq, scaler = prepare_input_sequence(df)
        print("Prepared Input Sequence")  # Debugging Line

        predictions = []
        for _ in range(3):  # Predict next 3 days
            pred = model(input_seq).cpu().detach().numpy()
            predictions.append(pred[0][0])
            

            # Append predicted value for next step
            new_seq = np.append(input_seq.cpu().numpy().squeeze(), pred)
            new_seq = new_seq[-60:].reshape(1, 60, 1)
            input_seq = torch.tensor(new_seq, dtype=torch.float32).to(device)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        df = df.dropna(subset=['Close'])  # Drop rows with NaN in 'Close' column
        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        print("Predicted Prices:", predicted_prices)  # Debugging Line
        pred = model(input_seq).cpu().detach().numpy()
        pred = np.nan_to_num(pred, nan=0.0)  # Replace NaNs with 0
        print("Model Prediction (with NaN handling):", pred)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Use a smaller learning rate
        return jsonify({
            "ticker": ticker,
            "predicted_prices": predicted_prices.tolist()
        })

    except Exception as e:
        print("Error:", str(e))  # Debugging Line
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


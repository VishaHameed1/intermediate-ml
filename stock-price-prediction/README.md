# Stock Price Prediction (Time Series)

## Overview
This project predicts future stock prices using **time-series forecasting**.  
Models used:
- **ARIMA** (Statistical)
- **LSTM** (Deep Learning)

Use cases:
- Financial forecasting
- Investment strategy
- Risk management

## Dataset
- Source: Yahoo Finance / Kaggle
- Columns:
  - `Date` – Date of stock price
  - `Open` – Opening price
  - `High` – Highest price
  - `Low` – Lowest price
  - `Close` – Closing price
  - `Volume` – Traded volume

## Key Steps
1. Load dataset and sort by date
2. Exploratory Data Analysis (EDA)
   - Check for missing values
   - Plot historical prices
3. Feature scaling (MinMaxScaler for LSTM)
4. Train/Test split (time-based)
5. Model building:
   - ARIMA → classical forecasting
   - LSTM → neural network forecasting
6. Model evaluation:
   - RMSE, MAE
   - Visualize predicted vs actual
7. Forecast future prices

## How to Run
```bash
python main.py
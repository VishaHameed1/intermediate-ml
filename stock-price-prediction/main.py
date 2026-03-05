import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1️⃣ Load dataset
try:
    df = pd.read_csv("stock-price-prediction/AAPL.csv", parse_dates=['Date'])
    print("Dataset loaded from file.")
except:
    print("Dataset not found. Creating synthetic dataset...")
    dates = pd.date_range(start='2020-01-01', periods=500)
    df = pd.DataFrame({
        'Date': dates,
        'Close': np.cumsum(np.random.randn(500)) + 100
    })

df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# 2️⃣ Feature scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_close = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# 3️⃣ Create sequences
look_back = 20
X, y = [], []

for i in range(look_back, len(scaled_close)):
    X.append(scaled_close[i-look_back:i,0])
    y.append(scaled_close[i,0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4️⃣ Train/Test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5️⃣ LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=32)

# 6️⃣ Prediction
predicted = model.predict(X_test)

predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# 7️⃣ Plot
plt.figure(figsize=(10,6))
plt.plot(actual_prices, label="Actual Price")
plt.plot(predicted_prices, label="Predicted Price")
plt.legend()
plt.show()
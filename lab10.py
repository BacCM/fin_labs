# ============================================================
# Лабораторная работа 10: MLP для прогнозирования цен акций
# ============================================================

# --------------------
# Этап 1. Сбор и первичная обработка данных
# --------------------
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Загрузка данных
ticker = "TSLA"
start_date = "2019-01-01"
end_date = "2024-01-01"

df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Close', 'Volume']]  # оставляем только нужные колонки

# Заполнение пропусков
df.fillna(method='ffill', inplace=True)
df.dropna(inplace=True)

print("Первые 5 строк:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())

# --------------------
# Этап 2. Feature Engineering
# --------------------
# Логарифмическая доходность
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Скользящие средние
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()

# Функция создания лаговых признаков
def create_lagged_features(df, window_size):
    df_lagged = df.copy()
    for i in range(1, window_size + 1):
        df_lagged[f'Close_Lag_{i}'] = df_lagged['Close'].shift(i)
        df_lagged[f'Volume_Lag_{i}'] = df_lagged['Volume'].shift(i)
    df_lagged['Target'] = df_lagged['Close'].shift(-1)  # целевая переменная
    df_lagged.dropna(inplace=True)
    return df_lagged

window_size = 10
df_lagged = create_lagged_features(df, window_size)

# Признаки и целевая переменная
feature_cols = [col for col in df_lagged.columns if col not in ['Close', 'Target']]
X = df_lagged[feature_cols].values
y = df_lagged['Target'].values

print(f"Размер X: {X.shape}, размер y: {y.shape}")

# --------------------
# Этап 3. Подготовка данных и масштабирование
# --------------------
# Разделение на train/test (последовательно)
split_idx = int(len(X) * 0.8)
X_train_full, X_test = X[:split_idx], X[split_idx:]
y_train_full, y_test = y[:split_idx], y[split_idx:]

# Разделение train на train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, shuffle=False
)

# Масштабирование
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# --------------------
# Этап 4. Построение и обучение модели MLP
# --------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Графики обучения
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('MAE')
plt.legend()
plt.show()

# --------------------
# Этап 5. Оценка модели и визуализация прогнозов
# --------------------
test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f"Тестовая MSE: {test_loss:.4f}, Тестовая MAE (масштабированная): {test_mae:.4f}")

# Прогноз и обратное масштабирование
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

# График реальных vs предсказанных цен
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Price', color='blue')
plt.plot(y_pred, label='Predicted Price', color='red', alpha=0.7)
plt.title('Прогноз цен закрытия TSLA')
plt.xlabel('Время (тестовые дни)')
plt.ylabel('Цена (USD)')
plt.legend()
plt.show()

# Итоговые метрики в долларах
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae_usd = mean_absolute_error(y_test_actual, y_pred)
rmse_usd = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print(f"MAE в долларах: ${mae_usd:.2f}")
print(f"RMSE в долларах: ${rmse_usd:.2f}")

# --------------------
# Сохранение модели и скейлеров (для этапа 6)
# --------------------
model.save("tsla_mlp_model.h5")
with open("scalers.pkl", "wb") as f:
    pickle.dump((scaler_X, scaler_y), f)

print("Модель и скейлеры сохранены.")
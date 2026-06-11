# ============================================================
# Лабораторная работа 10: MLP для прогнозирования цен акций
# ============================================================

import os
import warnings
import logging

# Подавление предупреждений TensorFlow и oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Отключаем логирование от ABSL
logging.getLogger('absl').setLevel(logging.ERROR)

# --------------------
# Этап 1. Сбор и первичная обработка данных
# --------------------
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

print("=" * 60)
print("Этап 1: Загрузка данных")
print("=" * 60)

# Загрузка данных
ticker = "TSLA"
start_date = "2019-01-01"
end_date = "2024-01-01"

try:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df = df[['Close', 'Volume']]  # оставляем только нужные колонки

    print(f"Загружено строк: {len(df)}")
    print(f"Период: с {df.index[0].date()} по {df.index[-1].date()}")

    # Заполнение пропусков
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    print("\nПервые 5 строк:")
    print(df.head())
    print("\nПоследние 5 строк:")
    print(df.tail())
    print(f"\nПропуски: {df.isnull().sum().sum()}")

except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    exit(1)

# --------------------
# Этап 2. Feature Engineering
# --------------------
print("\n" + "=" * 60)
print("Этап 2: Создание признаков")
print("=" * 60)

# Логарифмическая доходность
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Скользящие средние
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()

print(f"После добавления признаков: {df.shape}")


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

print(f"Размер X: {X.shape} (признаков: {X.shape[1]})")
print(f"Размер y: {y.shape}")
print(f"\nСписок признаков ({len(feature_cols)}):")
for i, col in enumerate(feature_cols[:10]):  # показываем первые 10
    print(f"  {i + 1}. {col}")
print(f"  ... и еще {len(feature_cols) - 10} признаков")

# --------------------
# Этап 3. Подготовка данных и масштабирование
# --------------------
print("\n" + "=" * 60)
print("Этап 3: Подготовка данных и масштабирование")
print("=" * 60)

# Разделение на train/test (последовательно)
split_idx = int(len(X) * 0.8)
X_train_full, X_test = X[:split_idx], X[split_idx:]
y_train_full, y_test = y[:split_idx], y[split_idx:]

print(f"Обучающая выборка (полная): {X_train_full.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# Разделение train на train/val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, shuffle=False
)

print(f"Тренировочная: {X_train.shape}")
print(f"Валидационная: {X_val.shape}")

# Масштабирование
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

print("\nМасштабирование выполнено")
print(f"Среднее X: {scaler_X.mean_[:3]}...")
print(f"Стандартное отклонение X: {scaler_X.scale_[:3]}...")

# --------------------
# Этап 4. Построение и обучение модели MLP
# --------------------
print("\n" + "=" * 60)
print("Этап 4: Построение и обучение модели MLP")
print("=" * 60)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Архитектура модели:")
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

print("\nНачало обучения...")
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
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Функция потерь (MSE)', fontsize=12)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
plt.title('Метрика MAE', fontsize=12)
plt.xlabel('Эпоха')
plt.ylabel('MAE')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100)
plt.show()

# --------------------
# Этап 5. Оценка модели и визуализация прогнозов
# --------------------
print("\n" + "=" * 60)
print("Этап 5: Оценка модели и визуализация")
print("=" * 60)

test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
print(f"Тестовая MSE (масштабированная): {test_loss:.4f}")
print(f"Тестовая MAE (масштабированная): {test_mae:.4f}")

# Прогноз и обратное масштабирование
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

# Статистика цен
print(f"\nСтатистика фактических цен (тест):")
print(f"  Минимум: ${y_test_actual.min():.2f}")
print(f"  Максимум: ${y_test_actual.max():.2f}")
print(f"  Среднее: ${y_test_actual.mean():.2f}")

# График реальных vs предсказанных цен
plt.figure(figsize=(15, 6))
plt.plot(y_test_actual, label='Фактическая цена', color='blue', linewidth=1.5)
plt.plot(y_pred, label='Предсказанная цена', color='red', alpha=0.7, linewidth=1.5)
plt.title('Прогноз цен закрытия TSLA (тестовая выборка)', fontsize=14)
plt.xlabel('Временные шаги (дни)', fontsize=12)
plt.ylabel('Цена (USD)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('price_prediction.png', dpi=100)
plt.show()

# Итоговые метрики в долларах
mae_usd = mean_absolute_error(y_test_actual, y_pred)
rmse_usd = np.sqrt(mean_squared_error(y_test_actual, y_pred))

print("\n" + "=" * 60)
print("ИТОГОВЫЕ МЕТРИКИ (в долларах США)")
print("=" * 60)
print(f"MAE (средняя абсолютная ошибка):  ${mae_usd:.2f}")
print(f"RMSE (среднеквадратичная ошибка): ${rmse_usd:.2f}")
print(f"MAPE (в процентах):              {(mae_usd / y_test_actual.mean() * 100):.2f}%")

# Дополнительная визуализация: график ошибок
errors = y_test_actual.flatten() - y_pred.flatten()
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Ошибка прогноза (USD)')
plt.ylabel('Частота')
plt.title('Распределение ошибок прогноза')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test_actual, y_pred, alpha=0.5, s=10)
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()],
         'r--', linewidth=2, label='Идеальный прогноз')
plt.xlabel('Фактическая цена (USD)')
plt.ylabel('Предсказанная цена (USD)')
plt.title('Фактические vs предсказанные значения')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('errors_analysis.png', dpi=100)
plt.show()

# --------------------
# Сохранение модели и скейлеров (для этапа 6)
# --------------------
print("\n" + "=" * 60)
print("Сохранение модели и скейлеров")
print("=" * 60)

model.save("tsla_mlp_model.h5")
with open("scalers.pkl", "wb") as f:
    pickle.dump((scaler_X, scaler_y, feature_cols), f)  # сохраняем также имена признаков

print("✓ Модель сохранена в файл: tsla_mlp_model.h5")
print("✓ Скейлеры сохранены в файл: scalers.pkl")
print(f"✓ Сохранено {len(feature_cols)} имен признаков")

print("\n" + "=" * 60)
print("Работа успешно завершена!")
print("=" * 60)
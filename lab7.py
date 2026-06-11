import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
import warnings

from IPython.display import display

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")

# ## 1. Загрузка и подготовка данных

df = pd.read_csv('covid_19_data.csv')

print("Исходная форма данных:")
print(df.shape)
print("\nТипы данных:")
print(df.dtypes)
print("\nПервые 5 строк:")
display(df.head())

# ## 2. Очистка и агрегация

# Проверим наличие пропусков
print("\nПропуски в данных:")
print(df.isnull().sum())

# Группируем по дате, суммируя случаи по всему миру (или можно по странам)
ts = df.groupby('ObservationDate')['Confirmed'].sum().reset_index()

# Преобразуем даты и сортируем
ts['ObservationDate'] = pd.to_datetime(ts['ObservationDate'])
ts = ts.sort_values('ObservationDate').reset_index(drop=True)

print("\nДанные после агрегации:")
display(ts.head())
print("\nДиапазон дат:", ts['ObservationDate'].min(), "—", ts['ObservationDate'].max())

# Переименуем для удобства работы с Prophet / Statsmodels
ts = ts.rename(columns={'ObservationDate': 'ds', 'Confirmed': 'y'})


# ## 3. Описательная статистика и визуализация

plt.figure(figsize=(14, 7))
plt.plot(ts['ds'], ts['y'], color='indianred', linewidth=1.5)
plt.title('Динамика подтвержденных случаев COVID-19 (ежедневная сумма)')
plt.xlabel('Дата')
plt.ylabel('Количество случаев')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

print("\nОписательная статистика:")
display(ts['y'].describe())


# ## 4. Проверка стационарности (ADF-тест)

result = adfuller(ts['y'])
print("\nРезультаты ADF-теста (проверка стационарности):")
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
if result[1] <= 0.05:
    print("Ряд стационарен (H0 отвергнута).")
else:
    print("Ряд не стационарен (H0 не отвергнута). Требуется дифференцирование.")

# ## 5. Декомпозиция временного ряда

decomposition = seasonal_decompose(ts.set_index('ds'), model='multiplicative', period=7)
fig = plt.figure(figsize=(12, 8))
fig = decomposition.plot()
plt.show()

# ## 6. Анализ автокорреляции

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ts['y'], ax=ax[0], lags=40)
plot_pacf(ts['y'], ax=ax[1], lags=40)
ax[0].set_title("Автокорреляционная функция (ACF)")
ax[1].set_title("Частная автокорреляционная функция (PACF)")
plt.tight_layout()
plt.show()

# ## 7. Построение прогноза с помощью Prophet

model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
model_prophet.fit(ts)
future = model_prophet.make_future_dataframe(periods=60) # Прогноз на 60 дней вперед
forecast = model_prophet.predict(future)

fig1 = model_prophet.plot(forecast)
plt.title('Прогноз Prophet: заболеваемость COVID-19')
plt.xlabel('Дата')
plt.ylabel('Количество случаев')
plt.show()

fig2 = model_prophet.plot_components(forecast)
plt.show()

# ## 8. Сравнение с моделью ARIMA

train_size = int(len(ts) * 0.8)
train, test = ts.iloc[:train_size], ts.iloc[train_size:]

print("\nПодбор оптимальной модели ARIMA...")
stepwise_model = auto_arima(train['y'], start_p=1, start_q=1,
                            max_p=3, max_q=3, m=7,
                            start_P=0, seasonal=True,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
print(f"Лучшая модель ARIMA: {stepwise_model.order} сезонная {stepwise_model.seasonal_order}")

stepwise_model.fit(train['y'])
future_forecast = stepwise_model.predict(n_periods=len(test))
future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])

plt.figure(figsize=(12, 6))
plt.plot(train['ds'], train['y'], label='Обучающая выборка')
plt.plot(test['ds'], test['y'], label='Тестовая выборка (факт)')
plt.plot(test['ds'], future_forecast, label='Прогноз ARIMA')
plt.title('Сравнение фактических данных и прогноза ARIMA')
plt.legend()
plt.show()
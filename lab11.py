# =====================================================
# ЛАБОРАТОРНАЯ РАБОТА №11 - ФИНАЛЬНАЯ ВЕРСИЯ
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import urllib.request
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings

warnings.filterwarnings('ignore')

# Настройка графиков
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# =====================================================
# ЭТАП 1. ПОДГОТОВКА ДАННЫХ И EDA
# =====================================================

print("=" * 60)
print("ЭТАП 1: ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
print("=" * 60)

# Загрузка датасета
print("\nЗагрузка данных...")

try:
    from sklearn.datasets import fetch_openml

    german_credit = fetch_openml(name='credit-g', version=1, as_frame=True)
    df = german_credit.frame
    print("✓ Данные загружены через fetch_openml")
except Exception as e:
    print(f"Загрузка через fetch_openml не удалась: {e}")
    print("Загрузка из GitHub...")

    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/german/german.data"
    column_names = [
        'checking_status', 'duration', 'credit_history', 'purpose', 'amount',
        'savings_status', 'employment', 'installment_commitment', 'personal_status',
        'other_parties', 'residence_since', 'property_magnitude', 'age',
        'other_payment_plans', 'housing', 'existing_credits', 'job',
        'num_dependents', 'own_telephone', 'foreign_worker', 'class'
    ]

    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8')
    lines = data.strip().split('\n')
    data_rows = [line.split() for line in lines]
    df = pd.DataFrame(data_rows, columns=column_names)

    # Преобразование целевой переменной
    df['class'] = df['class'].map({'1': 0, '2': 1})

    # Преобразование числовых столбцов
    numeric_cols = ['duration', 'amount', 'installment_commitment', 'residence_since',
                    'age', 'existing_credits', 'num_dependents']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    print("✓ Данные загружены из GitHub")

print(f"\nРазмер датасета: {df.shape}")
print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

print("\nПервые 5 строк:")
print(df.head())

print("\nТипы данных:")
print(df.dtypes)

print("\nОписательная статистика:")
print(df.describe())

# Проверка пропусков
print("\nПропуски:")
print(df.isnull().sum())

# Определение типов признаков
target = 'class'
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in numerical_cols:
    numerical_cols.remove(target)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nЧисловые признаки ({len(numerical_cols)}): {numerical_cols}")
print(f"Категориальные признаки ({len(categorical_cols)}): {categorical_cols[:5]}...")

# Визуализация целевой переменной
plt.figure(figsize=(8, 5))
class_counts = df[target].value_counts()
class_labels = ['Хороший (0)', 'Плохой (1)']
colors = ['#2ecc71', '#e74c3c']

bars = plt.bar(class_labels, class_counts.values, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Распределение целевой переменной', fontsize=14, fontweight='bold')
plt.xlabel('Класс', fontsize=12)
plt.ylabel('Количество клиентов', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Добавление значений
for bar, count in zip(bars, class_counts.values):
    percentage = count / len(df) * 100
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f'{count}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ИСПРАВЛЕНО: безопасный вывод баланса классов
print(f"\nБаланс классов:")
for idx, (class_value, count) in enumerate(class_counts.items()):
    percentage = count / len(df) * 100
    class_name = "Хорошие (0)" if class_value == 0 or class_value == '0' or idx == 0 else "Плохие (1)"
    print(f"{class_name}: {percentage:.2f}% ({count} клиентов)")

# Корреляционная матрица
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 8))
    corr_matrix = df[numerical_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, square=True, mask=mask, cbar_kws={"shrink": 0.8})
    plt.title('Корреляционная матрица числовых признаков', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# =====================================================
# ЭТАП 2. ПОСТРОЕНИЕ МОДЕЛЕЙ
# =====================================================

print("\n" + "=" * 60)
print("ЭТАП 2: ПОСТРОЕНИЕ И СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

# Подготовка данных
X = df.drop(columns=[target])
y = df[target].astype(int)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# Препроцессинг
if len(categorical_cols) > 0:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
else:
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_cols)]
    )

# Модели для обучения
models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'params': {'classifier__C': [0.01, 0.1, 1, 10]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5]
        }
    },
    'LGBM': {
        'model': LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, class_weight='balanced'),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.01, 0.1]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'),
        'params': {
            'classifier__iterations': [50, 100],
            'classifier__depth': [3, 6],
            'classifier__learning_rate': [0.01, 0.1]
        }
    }
}

# Обучение и оценка
results = []

for name, config in models.items():
    print(f"\n{'=' * 50}")
    print(f"Обучение: {name}")
    print('=' * 50)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])

    # Поиск лучших параметров
    grid_search = GridSearchCV(
        pipeline,
        param_grid=config['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"CV AUC-ROC: {grid_search.best_score_:.4f}")

    # Предсказания
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Метрики
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'AUC-ROC': roc_auc,
        'Model Object': best_model
    })

    print(f"\nРезультаты на тесте:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  AUC-ROC:   {roc_auc:.4f}")

# Сравнительная таблица
results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']]
print("\n" + "=" * 70)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC-ROC
colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = axes[0].bar(results_df['Model'], results_df['AUC-ROC'], color=colors_bar, edgecolor='black', linewidth=1.5)
axes[0].set_ylim(0, 1)
axes[0].set_title('Сравнение моделей по AUC-ROC', fontsize=14, fontweight='bold')
axes[0].set_ylabel('AUC-ROC', fontsize=12)
axes[0].set_xlabel('Модель', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

for bar, v in zip(bars, results_df['AUC-ROC']):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{v:.3f}', ha='center', fontweight='bold')

# Все метрики
results_melted = results_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
sns.barplot(data=results_melted, x='Model', y='Score', hue='Metric', ax=axes[1], palette='Set2')
axes[1].set_title('Сравнение всех метрик', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].legend(loc='lower right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================
# ЭТАП 3. SHAP АНАЛИЗ
# =====================================================

print("\n" + "=" * 60)
print("ЭТАП 3: SHAP АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ")
print("=" * 60)

# Выбор лучшей модели
best_idx = results_df['AUC-ROC'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_model_obj = results[best_idx]['Model Object']

print(f"\nЛучшая модель: {best_model_name}")
print(f"AUC-ROC: {results_df.loc[best_idx, 'AUC-ROC']:.4f}")

# SHAP анализ только для tree-based моделей
if best_model_name in ['RandomForest', 'LGBM', 'CatBoost']:
    print("\nВыполнение SHAP анализа...")

    # Получение названий признаков
    if len(categorical_cols) > 0:
        cat_encoder = best_model_obj.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    else:
        cat_feature_names = []

    feature_names_all = numerical_cols + cat_feature_names

    # Подготовка данных
    X_test_processed = best_model_obj.named_steps['preprocessor'].transform(X_test)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names_all)

    # Расчет SHAP
    explainer = shap.TreeExplainer(best_model_obj.named_steps['classifier'])
    shap_values = explainer.shap_values(X_test_df)

    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values

    # График важности признаков
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, X_test_df, plot_type="bar", show=False)
    plt.title(f'Важность признаков по SHAP - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Топ-5 признаков
    shap_importance = pd.DataFrame({
        'feature': feature_names_all,
        'importance': np.abs(shap_values_class1).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\nТОП-5 наиболее важных признаков:")
    print("-" * 50)
    for i, (_, row) in enumerate(shap_importance.head(5).iterrows(), 1):
        print(f"{i}. {row['feature']}: {row['importance']:.4f}")

    # Анализ конкретного клиента
    y_pred_proba = best_model_obj.predict_proba(X_test)[:, 1]

    # Поиск клиента с отказом
    bad_idx = np.where((y_pred_proba > 0.6) & (y_test == 1))[0]
    if len(bad_idx) > 0:
        client_idx = bad_idx[0]
    else:
        client_idx = np.argmax(y_pred_proba)

    client_proba = y_pred_proba[client_idx]
    client_true = y_test.iloc[client_idx]

    print(f"\nАнализ клиента с отказом:")
    print("-" * 50)
    print(f"Истинный класс: {'Плохой' if client_true == 1 else 'Хороший'}")
    print(f"Вероятность дефолта: {client_proba:.2%}")
    print(f"Решение: {'ОТКАЗ' if client_proba >= 0.5 else 'ОДОБРЕНИЕ'}")

    # Force plot
    client_data = X_test.iloc[[client_idx]]
    client_processed = best_model_obj.named_steps['preprocessor'].transform(client_data)
    client_df = pd.DataFrame(client_processed, columns=feature_names_all)
    client_shap = explainer.shap_values(client_df)

    if isinstance(client_shap, list):
        client_shap_values = client_shap[1][0]
    else:
        client_shap_values = client_shap[0]

    expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                             list) else explainer.expected_value

    shap.initjs()
    plt.figure(figsize=(20, 4))
    shap.force_plot(expected_val, client_shap_values, client_df, matplotlib=True, show=False)
    plt.title(f'Force Plot - Вероятность дефолта: {client_proba:.2%}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nФакторы, увеличивающие риск (топ-3):")
    shap_client = pd.DataFrame({
        'feature': feature_names_all,
        'shap_value': client_shap_values
    }).sort_values('shap_value', ascending=False)

    for _, row in shap_client.head(3).iterrows():
        print(f"  ↑ {row['feature']}: +{row['shap_value']:.4f}")

    print("\nФакторы, снижающие риск (топ-3):")
    for _, row in shap_client.tail(3).iterrows():
        print(f"  ↓ {row['feature']}: {row['shap_value']:.4f}")

else:
    print(f"\nSHAP анализ пропущен (модель {best_model_name} не является tree-based)")

# =====================================================
# ЭТАП 4. СОХРАНЕНИЕ МОДЕЛИ
# =====================================================

print("\n" + "=" * 60)
print("ЭТАП 4: СОХРАНЕНИЕ МОДЕЛИ И СОЗДАНИЕ API")
print("=" * 60)

# Сохранение модели
joblib.dump(best_model_obj, 'credit_scoring_model.pkl')
print("✓ Модель сохранена: credit_scoring_model.pkl")

# Сохранение метаданных
metadata = {
    'feature_names': numerical_cols + categorical_cols,
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'best_model_name': best_model_name,
    'auc_roc_score': float(results_df.loc[best_idx, 'AUC-ROC'])
}
joblib.dump(metadata, 'model_metadata.pkl')
print("✓ Метаданные сохранены: model_metadata.pkl")

# Создание API файла
api_code = '''# api.py - Credit Scoring API
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import uvicorn

print("Loading model...")
model = joblib.load('credit_scoring_model.pkl')
metadata = joblib.load('model_metadata.pkl')
print(f"Loaded: {metadata['best_model_name']} (AUC-ROC: {metadata['auc_roc_score']:.4f})")

app = FastAPI(title="Credit Scoring API", version="1.0.0")

class CreditRequest(BaseModel):
    data: Dict[str, Any]

class CreditResponse(BaseModel):
    prediction: int
    probability_default: float
    credit_decision: str
    risk_level: str

@app.get("/")
def root():
    return {"model": metadata['best_model_name'], "auc_roc": metadata['auc_roc_score']}

@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    try:
        input_df = pd.DataFrame([request.data])
        proba = model.predict_proba(input_df)[0, 1]
        prediction = 1 if proba >= 0.5 else 0

        if proba < 0.3:
            risk_level = "Low"
        elif proba < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return CreditResponse(
            prediction=prediction,
            probability_default=float(proba),
            credit_decision="Rejected" if prediction == 1 else "Approved",
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

with open('api.py', 'w', encoding='utf-8') as f:
    f.write(api_code)
print("✓ API файл создан: api.py")

print("\n" + "=" * 60)
print("РАБОТА УСПЕШНО ЗАВЕРШЕНА!")
print("=" * 60)
print("\nИНСТРУКЦИЯ ПО ЗАПУСКУ API:")
print("1. pip install fastapi uvicorn")
print("2. python api.py")
print("3. Открыть http://127.0.0.1:8000/docs")
print("=" * 60)
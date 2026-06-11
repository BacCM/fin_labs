# =====================================================
# ЛАБОРАТОРНАЯ РАБОТА №11 - С ОТОБРАЖЕНИЕМ ГРАФИКОВ
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import urllib.request
import json
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# Настройка графиков для интерактивного отображения
plt.ion()  # Включаем интерактивный режим
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# =====================================================
# ЭТАП 1. ПОДГОТОВКА ДАННЫХ И EDA
# =====================================================

print("=" * 60)
print("ЭТАП 1: ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
print("=" * 60)

# Загрузка датасета
print("\nЗагрузка данных...")

# Пытаемся загрузить данные
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/german/german.data"
column_names = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'amount',
    'savings_status', 'employment', 'installment_commitment', 'personal_status',
    'other_parties', 'residence_since', 'property_magnitude', 'age',
    'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker', 'class'
]

try:
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

    print("✓ Данные успешно загружены из GitHub")

    # Определяем категориальные признаки (все остальные)
    categorical_cols = [col for col in df.columns if col not in numeric_cols + ['class']]

except Exception as e:
    print(f"Ошибка загрузки: {e}")
    print("Создание демонстрационного датасета...")

    # Создаем демо-датасет
    np.random.seed(42)
    n_samples = 1000

    numeric_cols = ['duration', 'amount', 'age', 'installment_commitment',
                    'residence_since', 'existing_credits', 'num_dependents']

    categorical_cols = ['checking_status', 'credit_history', 'purpose', 'savings_status',
                        'employment', 'personal_status', 'other_parties', 'property_magnitude',
                        'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker']

    data_dict = {
        'duration': np.random.randint(4, 72, n_samples),
        'amount': np.random.randint(250, 20000, n_samples),
        'age': np.random.randint(19, 75, n_samples),
        'installment_commitment': np.random.randint(1, 4, n_samples),
        'residence_since': np.random.randint(1, 5, n_samples),
        'existing_credits': np.random.randint(1, 4, n_samples),
        'num_dependents': np.random.randint(1, 3, n_samples),
        'checking_status': np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples),
        'credit_history': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples),
        'purpose': np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'],
                                    n_samples),
        'savings_status': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n_samples),
        'employment': np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], n_samples),
        'personal_status': np.random.choice(['A91', 'A92', 'A93', 'A94', 'A95'], n_samples),
        'other_parties': np.random.choice(['A101', 'A102', 'A103'], n_samples),
        'property_magnitude': np.random.choice(['A121', 'A122', 'A123', 'A124'], n_samples),
        'other_payment_plans': np.random.choice(['A141', 'A142', 'A143'], n_samples),
        'housing': np.random.choice(['A151', 'A152', 'A153'], n_samples),
        'job': np.random.choice(['A171', 'A172', 'A173', 'A174'], n_samples),
        'own_telephone': np.random.choice(['A191', 'A192'], n_samples),
        'foreign_worker': np.random.choice(['A201', 'A202'], n_samples)
    }

    df = pd.DataFrame(data_dict)
    # Создаем целевую переменную с дисбалансом
    df['class'] = np.where(
        (df['duration'] > 36) | (df['amount'] > 10000) | (df['age'] < 25),
        np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    )

    print("✓ Создан демонстрационный датасет")

print(f"\nРазмер датасета: {df.shape}")
print(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

print("\nПервые 5 строк:")
print(df.head())

print(f"\nЧисловые признаки ({len(numeric_cols)}): {numeric_cols}")
print(f"Категориальные признаки ({len(categorical_cols)}): {categorical_cols[:5]}...")

# 1. Визуализация распределения целевой переменной
print("\n→ Отображение графика: Распределение целевой переменной")
plt.figure(figsize=(8, 5))
class_counts = df['class'].value_counts()
class_labels = ['Хороший (0)', 'Плохой (1)']
colors = ['#2ecc71', '#e74c3c']

bars = plt.bar(class_labels[:len(class_counts)], class_counts.values,
               color=colors[:len(class_counts)], edgecolor='black', linewidth=1.5)
plt.title('Распределение целевой переменной', fontsize=14, fontweight='bold')
plt.xlabel('Класс', fontsize=12)
plt.ylabel('Количество клиентов', fontsize=12)
plt.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, class_counts.values):
    percentage = count / len(df) * 100
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f'{count}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show(block=False)  # Показываем без блокировки
plt.savefig('target_distribution.png', dpi=100, bbox_inches='tight')
print("  ✓ График сохранен: target_distribution.png")
input("\nНажмите Enter для продолжения...")

# Баланс классов
print(f"\nБаланс классов:")
for i, (class_value, count) in enumerate(class_counts.items()):
    percentage = count / len(df) * 100
    class_name = "Хорошие (0)" if i == 0 else "Плохие (1)"
    print(f"  {class_name}: {percentage:.2f}% ({count} клиентов)")

# 2. Корреляционная матрица
if len(numeric_cols) > 1:
    print("\n→ Отображение графика: Корреляционная матрица")
    plt.figure(figsize=(12, 8))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, square=True, mask=mask, cbar_kws={"shrink": 0.8})
    plt.title('Корреляционная матрица числовых признаков', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig('correlation_matrix.png', dpi=100, bbox_inches='tight')
    print("  ✓ График сохранен: correlation_matrix.png")
    input("\nНажмите Enter для продолжения...")

# =====================================================
# ЭТАП 2. ПОСТРОЕНИЕ МОДЕЛЕЙ
# =====================================================

print("\n" + "=" * 60)
print("ЭТАП 2: ПОСТРОЕНИЕ И СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

# Подготовка данных
X = df.drop(columns=['class'])
y = df['class'].astype(int)

print(f"\nТип y: {y.dtype}")
print(f"Уникальные значения y: {y.unique()}")
print(f"Распределение y: {y.value_counts().to_dict()}")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Распределение в обучающей: {y_train.value_counts().to_dict()}")
print(f"Распределение в тестовой: {y_test.value_counts().to_dict()}")

# Препроцессинг
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
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

    try:
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

        print("  Поиск гиперпараметров...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"  Лучшие параметры: {grid_search.best_params_}")
        print(f"  CV AUC-ROC: {grid_search.best_score_:.4f}")

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

        print(f"\n  Результаты на тестовой выборке:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    AUC-ROC:   {roc_auc:.4f}")

    except Exception as e:
        print(f"  ✗ Ошибка при обучении {name}: {e}")
        continue

# Проверка результатов
if len(results) == 0:
    print("\n❌ Ни одна модель не была успешно обучена!")
    exit(1)

# Сравнительная таблица
results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']]
print("\n" + "=" * 70)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)

# Сохраняем таблицу
results_df.to_csv('model_comparison.csv', index=False)
print("\n✓ Таблица сохранена: model_comparison.csv")

# 3. Визуализация сравнения моделей
print("\n→ Отображение графика: Сравнение моделей")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC-ROC
colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = axes[0].bar(results_df['Model'], results_df['AUC-ROC'],
                   color=colors_bar[:len(results_df)], edgecolor='black', linewidth=1.5)
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
plt.show(block=False)
plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
print("✓ График сохранен: model_comparison.png")
input("\nНажмите Enter для продолжения...")

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

# SHAP анализ для tree-based моделей
if best_model_name in ['RandomForest', 'LGBM', 'CatBoost']:
    print("\nВыполнение SHAP анализа...")

    try:
        # Получение названий признаков
        cat_encoder = best_model_obj.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()

        feature_names_all = numeric_cols + cat_feature_names

        # Подготовка данных
        X_test_processed = best_model_obj.named_steps['preprocessor'].transform(X_test)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_names_all)

        # Расчет SHAP
        print("  Расчет SHAP значений (может занять некоторое время)...")
        explainer = shap.TreeExplainer(best_model_obj.named_steps['classifier'])
        shap_values = explainer.shap_values(X_test_df)

        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        else:
            shap_values_class1 = shap_values

        # 4. График важности признаков
        print("\n→ Отображение графика: Важность признаков (SHAP)")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_class1, X_test_df, plot_type="bar", show=False)
        plt.title(f'Важность признаков по SHAP - {best_model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig('shap_importance.png', dpi=100, bbox_inches='tight')
        print("  ✓ График сохранен: shap_importance.png")

        # Топ-5 признаков
        shap_importance = pd.DataFrame({
            'feature': feature_names_all,
            'importance': np.abs(shap_values_class1).mean(axis=0)
        }).sort_values('importance', ascending=False)

        print("\n  ТОП-5 наиболее важных признаков:")
        print("  " + "-" * 50)
        for i, (_, row) in enumerate(shap_importance.head(5).iterrows(), 1):
            print(f"  {i}. {row['feature']}: {row['importance']:.4f}")

        # Сохраняем важность признаков
        shap_importance.to_csv('shap_feature_importance.csv', index=False)
        print("  ✓ shap_feature_importance.csv")

        # 5. Force plot для конкретного клиента
        print("\n→ Анализ конкретного клиента")
        y_pred_proba = best_model_obj.predict_proba(X_test)[:, 1]

        # Находим клиента с отказом
        bad_indices = np.where((y_pred_proba > 0.6) & (y_test == 1))[0]
        if len(bad_indices) > 0:
            client_idx = bad_indices[0]
        else:
            client_idx = np.argmax(y_pred_proba)

        client_proba = y_pred_proba[client_idx]
        client_true = y_test.iloc[client_idx]

        print(f"\n  Выбран клиент #{client_idx}")
        print(f"  Истинный класс: {'Плохой' if client_true == 1 else 'Хороший'}")
        print(f"  Вероятность дефолта: {client_proba:.2%}")
        print(f"  Решение: {'ОТКАЗАТЬ' if client_proba >= 0.5 else 'ОДОБРИТЬ'}")

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

        print("\n  → Отображение Force Plot (может открыться в новом окне)")
        shap.initjs()
        force_plot = shap.force_plot(expected_val, client_shap_values, client_df, matplotlib=True)

        # Сохраняем force plot
        plt.figure(figsize=(20, 4))
        shap.force_plot(expected_val, client_shap_values, client_df, matplotlib=True, show=False)
        plt.title(f'Force Plot - Вероятность дефолта: {client_proba:.2%}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('force_plot.png', dpi=100, bbox_inches='tight')
        plt.show(block=False)
        print("  ✓ Force Plot сохранен: force_plot.png")

        print("\n  Факторы, увеличивающие риск (топ-3):")
        shap_client = pd.DataFrame({
            'feature': feature_names_all,
            'shap_value': client_shap_values
        }).sort_values('shap_value', ascending=False)

        for _, row in shap_client.head(3).iterrows():
            print(f"    ↑ {row['feature']}: +{row['shap_value']:.4f}")

        print("\n  Факторы, снижающие риск (топ-3):")
        for _, row in shap_client.tail(3).iterrows():
            print(f"    ↓ {row['feature']}: {row['shap_value']:.4f}")

        input("\nНажмите Enter для продолжения...")

    except Exception as e:
        print(f"  ✗ Ошибка при SHAP анализе: {e}")
else:
    print(f"\nSHAP анализ пропущен (модель {best_model_name} не является tree-based)")

# =====================================================
# ЭТАП 4. СОХРАНЕНИЕ МОДЕЛИ
# =====================================================

print("\n" + "=" * 60)
print("ЭТАП 4: СОХРАНЕНИЕ МОДЕЛИ И СОЗДАНИЕ API")
print("=" * 60)

try:
    # Сохранение модели
    joblib.dump(best_model_obj, 'credit_scoring_model.pkl')
    print("✓ Модель сохранена: credit_scoring_model.pkl")

    # Сохранение метаданных
    metadata = {
        'feature_names': numeric_cols + categorical_cols,
        'categorical_cols': categorical_cols,
        'numerical_cols': numeric_cols,
        'best_model_name': best_model_name,
        'auc_roc_score': float(results_df.loc[best_idx, 'AUC-ROC'])
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    print("✓ Метаданные сохранены: model_metadata.pkl")

    # Сохранение препроцессора отдельно
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("✓ Препроцессор сохранен: preprocessor.pkl")

    # Создание API файла
    api_code = '''# api.py - Credit Scoring API
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import uvicorn

# Загрузка модели и компонентов
print("Loading model components...")
model = joblib.load('credit_scoring_model.pkl')
metadata = joblib.load('model_metadata.pkl')
preprocessor = joblib.load('preprocessor.pkl')

print(f"✓ Model loaded: {metadata['best_model_name']}")
print(f"✓ AUC-ROC: {metadata['auc_roc_score']:.4f}")
print(f"✓ Features: {len(metadata['feature_names'])}")

app = FastAPI(
    title="Credit Scoring API",
    description="API для предсказания вероятности дефолта по кредиту",
    version="1.0.0"
)

class CreditRequest(BaseModel):
    """Запрос на оценку кредитного риска"""
    data: Dict[str, Any] = Field(..., description="Признаки клиента")

    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "duration": 6,
                    "amount": 1169,
                    "age": 67,
                    "checking_status": "A11"
                }
            }
        }

class CreditResponse(BaseModel):
    """Ответ сервиса"""
    prediction: int = Field(..., description="0 - одобрить, 1 - отказать")
    probability_default: float = Field(..., description="Вероятность дефолта (0-1)")
    credit_decision: str = Field(..., description="Решение по кредиту")
    risk_level: str = Field(..., description="Уровень риска: Low/Medium/High")

@app.get("/")
def root():
    """Информация о сервисе"""
    return {
        "service": "Credit Scoring API",
        "model": metadata['best_model_name'],
        "auc_roc": metadata['auc_roc_score'],
        "status": "active"
    }

@app.get("/health")
def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy"}

@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    """
    Предсказание вероятности дефолта

    Принимает JSON с признаками клиента, возвращает решение по кредиту
    """
    try:
        # Преобразование в DataFrame
        input_df = pd.DataFrame([request.data])

        # Предсказание вероятности
        proba = model.predict_proba(input_df)[0, 1]
        prediction = 1 if proba >= 0.5 else 0

        # Определение уровня риска
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

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
'''

    with open('api.py', 'w', encoding='utf-8') as f:
        f.write(api_code)
    print("✓ API файл создан: api.py")

    # Создание тестового запроса
    test_request = {
        "data": {
            "duration": 6,
            "amount": 1169,
            "age": 67
        }
    }

    if len(categorical_cols) > 0:
        test_request["data"][categorical_cols[0]] = "A11"

    with open('test_request.json', 'w', encoding='utf-8') as f:
        json.dump(test_request, f, indent=2, ensure_ascii=False)
    print("✓ Тестовый запрос сохранен: test_request.json")

except Exception as e:
    print(f"✗ Ошибка при сохранении: {e}")

print("\n" + "=" * 60)
print("✅ РАБОТА УСПЕШНО ЗАВЕРШЕНА!")
print("=" * 60)

print("\n📊 РЕЗУЛЬТАТЫ:")
print("-" * 60)
print(f"Лучшая модель: {best_model_name}")
print(f"AUC-ROC: {results_df.loc[best_idx, 'AUC-ROC']:.4f}")

print("\n🚀 ИНСТРУКЦИЯ ПО ЗАПУСКУ API:")
print("-" * 60)
print("1. Установите дополнительные библиотеки:")
print("   pip install fastapi uvicorn")
print("\n2. Запустите сервис:")
print("   python api.py")
print("\n3. Откройте документацию API:")
print("   http://127.0.0.1:8000/docs")
print("=" * 60)

# Держим окна открытыми
plt.show(block=True)
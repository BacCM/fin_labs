# =====================================================
# ЛАБОРАТОРНАЯ РАБОТА №11 - ИСПРАВЛЕННАЯ ВЕРСИЯ
# Тема: Сравнительный анализ архитектур для бинарной классификации
# Датасет: German Credit Data (альтернативная загрузка)
# =====================================================

# =====================================================
# 1. ИМПОРТ БИБЛИОТЕК
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import urllib.request
import io

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

# 1.1 Альтернативная загрузка датасета German Credit Data
print("Загрузка данных...")

# Способ 1: Попытка загрузить через fetch_openml с корректным id
try:
    from sklearn.datasets import fetch_openml

    german_credit = fetch_openml(data_id=31, as_frame=True)  # data_id=31 для German Credit
    df = german_credit.frame
    print("Данные загружены через fetch_openml с data_id=31")
except Exception as e:
    print(f"Ошибка загрузки через fetch_openml: {e}")
    print("Загрузка из локального файла...")

    # Способ 2: Прямая загрузка с GitHub
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/german/german.data"
    column_names = [
        'checking_status', 'duration', 'credit_history', 'purpose', 'amount',
        'savings_status', 'employment', 'installment_commitment', 'personal_status',
        'other_parties', 'residence_since', 'property_magnitude', 'age',
        'other_payment_plans', 'housing', 'existing_credits', 'job',
        'num_dependents', 'own_telephone', 'foreign_worker', 'class'
    ]

    try:
        # Загрузка данных
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
        lines = data.strip().split('\n')
        data_rows = [line.split() for line in lines]
        df = pd.DataFrame(data_rows, columns=column_names)

        # Преобразование целевой переменной: '1' -> 0 (good), '2' -> 1 (bad)
        df['class'] = df['class'].map({'1': 0, '2': 1})

        # Преобразование числовых столбцов
        numeric_cols = ['duration', 'amount', 'installment_commitment', 'residence_since',
                        'age', 'existing_credits', 'num_dependents']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        print("Данные успешно загружены из GitHub!")
    except Exception as e2:
        print(f"Ошибка загрузки из GitHub: {e2}")
        print("Использование встроенного датасета из sklearn...")

        # Способ 3: Использование make_classification (синтетические данные)
        from sklearn.datasets import make_classification

        X_synthetic, y_synthetic = make_classification(
            n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
            n_classes=2, weights=[0.7, 0.3], random_state=42
        )

        feature_names_synthetic = [f'feature_{i}' for i in range(20)]
        df = pd.DataFrame(X_synthetic, columns=feature_names_synthetic)
        df['class'] = y_synthetic
        print("Создан синтетический датасет (т.к. оригинальный недоступен)")

print(f"\nРазмер датасета: {df.shape}")
print("\nПервые 5 строк данных:")
print(df.head(), "\n")

print("Информация о данных:")
print(df.info(), "\n")

print("Описательная статистика:")
print(df.describe(), "\n")

# 1.3 Проверка пропусков
print("Пропуски в каждом столбце:")
print(df.isnull().sum(), "\n")

# 1.4 Определение числовых и категориальных признаков
target = 'class'

# Автоматическое определение типов
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in numerical_cols:
    numerical_cols.remove(target)

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Если нет категориальных признаков (синтетические данные), создаем разделение
if len(categorical_cols) == 0 and len(numerical_cols) > 0:
    # Для синтетических данных используем первые 15 как числовые, остальные 5 как категориальные
    if len(numerical_cols) >= 20:
        categorical_cols = numerical_cols[15:20]
        numerical_cols = numerical_cols[0:15]

print(f"Числовые признаки ({len(numerical_cols)}): {numerical_cols[:5] if len(numerical_cols) > 5 else numerical_cols}")
print(
    f"Категориальные признаки ({len(categorical_cols)}): {categorical_cols[:5] if len(categorical_cols) > 5 else categorical_cols}\n")

# 1.5 Визуализация целевой переменной (баланс классов)
plt.figure(figsize=(8, 5))
class_counts = df[target].value_counts()
colors = ['green', 'red']
bars = plt.bar(['Хороший (0)', 'Плохой (1)'], class_counts.values, color=colors)
plt.title('Распределение целевой переменной', fontsize=14, fontweight='bold')
plt.xlabel('Класс', fontsize=12)
plt.ylabel('Количество', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Добавление значений на столбцы
for bar, count in zip(bars, class_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f'{count} ({count / len(df) * 100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Оценка баланса
class_percent = df[target].value_counts(normalize=True)
print(f"\nБаланс классов:")
print(f"Хорошие (0): {class_percent[0]:.2%}")
print(f"Плохие (1): {class_percent[1]:.2%}\n")

# 1.6 Корреляционная матрица для числовых признаков (если есть)
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 8))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, square=True, mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
    plt.title('Корреляционная матрица числовых признаков', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("Недостаточно числовых признаков для построения корреляционной матрицы")

# =====================================================
# ЭТАП 2. ПОСТРОЕНИЕ МОДЕЛЕЙ И СРАВНИТЕЛЬНЫЙ АНАЛИЗ
# =====================================================

# 2.1 Разделение на обучающую (80%) и тестовую (20%)
X = df.drop(columns=[target])
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}\n")

# 2.2 Создание конвейера предобработки
# Обработка категориальных признаков (если есть)
if len(categorical_cols) > 0:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols)
        ]
    )

# 2.3 Определение моделей и параметров для GridSearchCV
models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'classifier__C': [0.01, 0.1, 1, 10]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5]
        }
    },
    'LGBM': {
        'model': LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.01, 0.1]
        }
    },
    'CatBoost': {
        'model': CatBoostClassifier(random_state=42, verbose=0),
        'params': {
            'classifier__iterations': [50, 100],
            'classifier__depth': [3, 6],
            'classifier__learning_rate': [0.01, 0.1]
        }
    }
}

# Результаты будем сохранять в список
results = []

for name, config in models.items():
    print(f"\n{'=' * 50}")
    print(f"Обучаем {name}...")
    print('=' * 50)

    # Создаем полный пайплайн
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])

    # Подбор гиперпараметров с кросс-валидацией
    grid_search = GridSearchCV(
        pipeline,
        param_grid=config['params'],
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Лучшая модель
    best_model = grid_search.best_estimator_
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшее CV AUC-ROC: {grid_search.best_score_:.4f}")

    # Предсказания на тесте
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

    print(f"\nРезультаты на тестовой выборке:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")

# 2.6 Сравнительная таблица результатов
results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']]
print("\n" + "=" * 70)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)

# Визуализация сравнения моделей
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График AUC-ROC
axes[0].bar(results_df['Model'], results_df['AUC-ROC'], color='skyblue', edgecolor='navy', linewidth=2)
axes[0].set_ylim(0, 1)
axes[0].set_title('Сравнение моделей по AUC-ROC', fontsize=14, fontweight='bold')
axes[0].set_ylabel('AUC-ROC', fontsize=12)
axes[0].set_xlabel('Модель', fontsize=12)
for i, v in enumerate(results_df['AUC-ROC']):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

# График всех метрик
results_melted = results_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
sns.barplot(data=results_melted, x='Model', y='Score', hue='Metric', ax=axes[1])
axes[1].set_title('Сравнение всех метрик', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].legend(loc='lower right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================
# ЭТАП 3. ИНТЕРПРЕТАЦИЯ МОДЕЛИ И АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
# =====================================================

# 3.1 Выбираем лучшую модель по AUC-ROC
best_model_name = results_df.loc[results_df['AUC-ROC'].idxmax(), 'Model']
best_model_obj = [res['Model Object'] for res in results if res['Model'] == best_model_name][0]
print(f"\n{'=' * 50}")
print(f"Лучшая модель по AUC-ROC: {best_model_name} (AUC-ROC = {results_df['AUC-ROC'].max():.4f})")
print('=' * 50)

# 3.2 Получение названий признаков после предобработки
try:
    # Получаем названия признаков после one-hot кодирования
    if len(categorical_cols) > 0:
        cat_encoder = best_model_obj.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    else:
        cat_feature_names = []

    feature_names = numerical_cols + cat_feature_names
    print(f"Количество признаков после предобработки: {len(feature_names)}")

    # Подготавливаем данные для SHAP
    X_test_processed = best_model_obj.named_steps['preprocessor'].transform(X_test)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    # 3.3 SHAP анализ для лучшей модели
    print("\nРасчет SHAP значений...")

    if best_model_name in ['RandomForest', 'LGBM', 'CatBoost']:
        # Для деревьев используем TreeExplainer
        explainer = shap.TreeExplainer(best_model_obj.named_steps['classifier'])
        shap_values = explainer.shap_values(X_test_df)

        # Для бинарной классификации берем значения для класса 1
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        else:
            shap_values_class1 = shap_values

        # Summary plot (bar)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_class1, X_test_df, plot_type="bar", show=False)
        plt.title(f"SHAP важность признаков - {best_model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Топ-5 признаков
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values_class1).mean(axis=0)
        }).sort_values('importance', ascending=False)

        print("\nТОП-5 наиболее важных признаков (по SHAP):")
        print("=" * 50)
        for idx, row in shap_importance.head(5).iterrows():
            print(f"{idx + 1}. {row['feature']}: {row['importance']:.4f}")

        # 3.4 Выбор клиента с отказом
        y_pred_proba_best = best_model_obj.predict_proba(X_test)[:, 1]
        bad_indices = np.where((y_pred_proba_best > 0.6) & (y_test == 1))[0]

        if len(bad_indices) > 0:
            chosen_idx = bad_indices[0]
        else:
            chosen_idx = np.argmax(y_pred_proba_best)

        chosen_client_data = X_test.iloc[[chosen_idx]]
        chosen_proba = y_pred_proba_best[chosen_idx]
        chosen_true = y_test.iloc[chosen_idx]

        print(f"\n{'=' * 50}")
        print(f"АНАЛИЗ КЛИЕНТА С ОТКАЗОМ")
        print('=' * 50)
        print(f"Индекс клиента: {chosen_idx}")
        print(f"Истинный класс: {'Плохой (дефолт)' if chosen_true == 1 else 'Хороший (без дефолта)'}")
        print(f"Предсказанная вероятность дефолта: {chosen_proba:.4f}")
        print(f"Решение модели: {'ОТКАЗАТЬ' if chosen_proba >= 0.5 else 'ОДОБРИТЬ'}")

        # Force plot для выбранного клиента
        print("\nПостроение Force Plot...")
        chosen_processed = best_model_obj.named_steps['preprocessor'].transform(chosen_client_data)
        chosen_df = pd.DataFrame(chosen_processed, columns=feature_names)
        chosen_shap = explainer.shap_values(chosen_df)

        if isinstance(chosen_shap, list):
            chosen_shap_values = chosen_shap[1][0]
        else:
            chosen_shap_values = chosen_shap[0]

        # Force plot
        shap.initjs()
        plt.figure(figsize=(20, 3))
        shap.force_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                        else explainer.expected_value,
                        chosen_shap_values, chosen_df, matplotlib=True, show=False)
        plt.title(f"Force Plot для клиента (вероятность дефолта: {chosen_proba:.2%})",
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Интерпретация
        print("\nИНТЕРПРЕТАЦИЯ РЕШЕНИЯ:")
        print("-" * 50)
        print("Факторы, увеличивающие риск дефолта (красные/положительные SHAP значения):")
        shap_client = pd.DataFrame({
            'feature': feature_names,
            'shap_value': chosen_shap_values
        }).sort_values('shap_value', ascending=False)

        print(shap_client.head(3).to_string(index=False))

        print("\nФакторы, снижающие риск дефолта (синие/отрицательные SHAP значения):")
        print(shap_client.tail(3).to_string(index=False))

    else:
        print(f"Для модели {best_model_name} SHAP анализ пропущен (требуется tree-based модель)")

except Exception as e:
    print(f"Ошибка при SHAP анализе: {e}")
    print("Пропускаем этот этап...")

# =====================================================
# ЭТАП 4. ДЕПЛОЙ МОДЕЛИ (БОНУСНЫЙ ЭТАП)
# =====================================================

# 4.1 Сохраняем лучшую модель
print("\n" + "=" * 50)
print("СОХРАНЕНИЕ МОДЕЛИ ДЛЯ API")
print("=" * 50)

try:
    joblib.dump(best_model_obj, 'credit_scoring_model.pkl')
    print("✓ Модель сохранена как 'credit_scoring_model.pkl'")

    # Сохраняем метаданные
    metadata = {
        'feature_names': numerical_cols + categorical_cols,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'best_model_name': best_model_name,
        'auc_roc_score': float(results_df['AUC-ROC'].max())
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    print("✓ Метаданные сохранены как 'model_metadata.pkl'")

    print("\n" + "=" * 50)
    print("API ФАЙЛ СОЗДАН")
    print("=" * 50)

    # Создание api.py файла
    api_code = '''# api.py - Веб-сервис для кредитного скоринга
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import uvicorn

# Загрузка модели и метаданных
print("Загрузка модели...")
model = joblib.load('credit_scoring_model.pkl')
metadata = joblib.load('model_metadata.pkl')
print(f"Модель загружена: {metadata['best_model_name']}")
print(f"AUC-ROC: {metadata['auc_roc_score']:.4f}")

app = FastAPI(
    title="Credit Scoring API",
    description="API для предсказания вероятности дефолта по кредитной заявке",
    version="1.0.0"
)

class CreditRequest(BaseModel):
    """Модель входных данных"""
    data: Dict[str, Any] = Field(..., description="Признаки клиента")

class CreditResponse(BaseModel):
    """Модель выходных данных"""
    prediction: int = Field(..., description="Предсказание: 0 - одобрить, 1 - отказать")
    probability_default: float = Field(..., description="Вероятность дефолта (0-1)")
    credit_decision: str = Field(..., description="Решение по кредиту")
    risk_level: str = Field(..., description="Уровень риска")

@app.get("/")
def root():
    return {
        "message": "Credit Scoring API is running",
        "model": metadata['best_model_name'],
        "auc_roc": metadata['auc_roc_score']
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    """
    Предсказание вероятности дефолта по кредитной заявке

    Пример тела запроса:
    {
        "data": {
            "duration": 6,
            "amount": 1169,
            "age": 67,
            "checking_status": "no checking"
        }
    }
    """
    try:
        # Преобразуем входные данные в DataFrame
        input_df = pd.DataFrame([request.data])

        # Проверка наличия всех необходимых признаков
        expected_features = metadata['feature_names']
        missing_cols = set(expected_features) - set(input_df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Отсутствуют обязательные поля: {list(missing_cols)}"
            )

        # Предсказание вероятности
        proba = model.predict_proba(input_df)[0, 1]
        prediction = 1 if proba >= 0.5 else 0

        # Определение уровня риска
        if proba < 0.3:
            risk_level = "Низкий"
        elif proba < 0.6:
            risk_level = "Средний"
        else:
            risk_level = "Высокий"

        return CreditResponse(
            prediction=prediction,
            probability_default=float(proba),
            credit_decision="Rejected" if prediction == 1 else "Approved",
            risk_level=risk_level
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    with open('api.py', 'w', encoding='utf-8') as f:
        f.write(api_code)
    print("✓ Файл 'api.py' создан успешно")

    # Создание тестового клиента
    test_client = {
        "data": {
            "duration": 6,
            "amount": 1169,
            "age": 67
        }
    }

    # Добавляем тестовые данные если есть категориальные признаки
    if len(categorical_cols) > 0:
        test_client["data"][categorical_cols[0]] = "unknown"

    with open('test_request.json', 'w', encoding='utf-8') as f:
        import json

        json.dump(test_client, f, indent=2, ensure_ascii=False)
    print("✓ Тестовый запрос сохранен в 'test_request.json'")

except Exception as e:
    print(f"✗ Ошибка при сохранении: {e}")

print("\n" + "=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
print("=" * 60)
print("\n📋 ИНСТРУКЦИЯ ПО ЗАПУСКУ API-СЕРВИСА:")
print("-" * 60)
print("1. Установите дополнительные библиотеки:")
print("   pip install fastapi uvicorn")
print("\n2. Запустите сервис командой:")
print("   python api.py")# =====================================================
# ЛАБОРАТОРНАЯ РАБОТА №11 - ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ВЕРСИЯ
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import urllib.request
import io
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# Настройка графиков
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# =====================================================
# ЭТАП 1. ПОДГОТОВКА ДАННЫХ И EDA
# =====================================================

print("="*60)
print("ЭТАП 1: ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
print("="*60)

# Загрузка датасета
print("\nЗагрузка данных...")

try:
    # Пробуем загрузить через fetch_openml
    from sklearn.datasets import fetch_openml
    german_credit = fetch_openml(name='credit-g', version=1, as_frame=True)
    df = german_credit.frame
    print("✓ Данные загружены через fetch_openml")
except Exception as e:
    print(f"Загрузка через fetch_openml не удалась: {e}")
    print("Загрузка из CSV файла...")

    # Альтернативная загрузка
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
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")

print("\nПервые 5 строк данных:")
print(df.head())

print("\nИнформация о данных:")
print(df.info())

print("\nОписательная статистика для числовых признаков:")
print(df.describe())

# Проверка пропусков
print("\nПропуски в каждом столбце:")
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
colors = ['#2ecc71', '#e74c3c']
bars = plt.bar(['Хороший (0)', 'Плохой (1)'], class_counts.values, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Распределение целевой переменной', fontsize=14, fontweight='bold')
plt.xlabel('Класс', fontsize=12)
plt.ylabel('Количество клиентов', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Добавление значений на столбцы
for bar, count in zip(bars, class_counts.values):
    percentage = count / len(df) * 100
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{count}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()

# Оценка баланса классов (ИСПРАВЛЕНО)
class_percent = df[target].value_counts(normalize=True)
print(f"\nБаланс классов:")
print(f"Хорошие (0): {class_percent.iloc[0]:.2%} ({class_counts.iloc[0]} клиентов)")
print(f"Плохие (1): {class_percent.iloc[1]:.2%} ({class_counts.iloc[1]} клиентов)")

# Корреляционная матрица для числовых признаков
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
# ЭТАП 2. ПОСТРОЕНИЕ МОДЕЛЕЙ И СРАВНИТЕЛЬНЫЙ АНАЛИЗ
# =====================================================

print("\n" + "="*60)
print("ЭТАП 2: ПОСТРОЕНИЕ И СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*60)

# Разделение данных
X = df.drop(columns=[target])
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Распределение в обучающей: {y_train.value_counts().to_dict()}")
print(f"Распределение в тестовой: {y_test.value_counts().to_dict()}")

# Создание конвейера предобработки
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

# Определение моделей
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

# Обучение моделей
results = []

for name, config in models.items():
    print(f"\n{'='*50}")
    print(f"Обучение модели: {name}")
    print('='*50)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])

    # Поиск гиперпараметров
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
    print(f"Лучшее CV AUC-ROC: {grid_search.best_score_:.4f}")

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

    print(f"\nРезультаты на тестовой выборке:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  AUC-ROC:   {roc_auc:.4f}")

# Сравнительная таблица
results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']]
print("\n" + "="*70)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)

# Визуализация сравнения
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График AUC-ROC
colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
bars = axes[0].bar(results_df['Model'], results_df['AUC-ROC'], color=colors_bar, edgecolor='black', linewidth=1.5)
axes[0].set_ylim(0, 1)
axes[0].set_title('Сравнение моделей по AUC-ROC', fontsize=14, fontweight='bold')
axes[0].set_ylabel('AUC-ROC', fontsize=12)
axes[0].set_xlabel('Модель', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

for bar, v in zip(bars, results_df['AUC-ROC']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)

# График всех метрик
results_melted = results_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
sns.barplot(data=results_melted, x='Model', y='Score', hue='Metric', ax=axes[1], palette='Set2')
axes[1].set_title('Сравнение всех метрик', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].legend(loc='lower right', fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================
# ЭТАП 3. ИНТЕРПРЕТАЦИЯ МОДЕЛИ
# =====================================================

print("\n" + "="*60)
print("ЭТАП 3: ИНТЕРПРЕТАЦИЯ МОДЕЛИ С SHAP")
print("="*60)

# Выбор лучшей модели
best_model_name = results_df.loc[results_df['AUC-ROC'].idxmax(), 'Model']
best_model_obj = [res['Model Object'] for res in results if res['Model'] == best_model_name][0]

print(f"\nЛучшая модель: {best_model_name}")
print(f"AUC-ROC: {results_df['AUC-ROC'].max():.4f}")

# SHAP анализ (только для tree-based моделей)
if best_model_name in ['RandomForest', 'LGBM', 'CatBoost']:
    print("\nВыполнение SHAP анализа...")

    # Получение названий признаков
    if len(categorical_cols) > 0:
        cat_encoder = best_model_obj.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    else:
        cat_feature_names = []

    feature_names = numerical_cols + cat_feature_names
    X_test_processed = best_model_obj.named_steps['preprocessor'].transform(X_test)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    # SHAP значения
    explainer = shap.TreeExplainer(best_model_obj.named_steps['classifier'])
    shap_values = explainer.shap_values(X_test_df)

    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values

    # График важности признаков
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class1, X_test_df, plot_type="bar", show=False)
    plt.title(f'Важность признаков (SHAP) - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Топ-5 признаков
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values_class1).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\nТОП-5 наиболее важных признаков:")
    print("-"*40)
    for idx, (_, row) in enumerate(shap_importance.head(5).iterrows(), 1):
        print(f"{idx}. {row['feature']}: {row['importance']:.4f}")

    # Анализ конкретного клиента
    y_pred_proba_best = best_model_obj.predict_proba(X_test)[:, 1]
    bad_indices = np.where((y_pred_proba_best > 0.6) & (y_test == 1))[0]

    if len(bad_indices) > 0:
        chosen_idx = bad_indices[0]
    else:
        chosen_idx = np.argmax(y_pred_proba_best)

    chosen_client = X_test.iloc[[chosen_idx]]
    chosen_proba = y_pred_proba_best[chosen_idx]
    chosen_true = y_test.iloc[chosen_idx]

    print(f"\nАнализ клиента с отказом:")
    print("-"*40)
    print(f"Индекс: {chosen_idx}")
    print(f"Истинный класс: {'Плохой' if chosen_true==1 else 'Хороший'}")
    print(f"Вероятность дефолта: {chosen_proba:.2%}")
    print(f"Решение: {'ОТКАЗАТЬ' if chosen_proba >= 0.5 else 'ОДОБРИТЬ'}")

    # Force plot
    chosen_processed = best_model_obj.named_steps['preprocessor'].transform(chosen_client)
    chosen_df = pd.DataFrame(chosen_processed, columns=feature_names)
    chosen_shap = explainer.shap_values(chosen_df)

    if isinstance(chosen_shap, list):
        chosen_shap_values = chosen_shap[1][0]
    else:
        chosen_shap_values = chosen_shap[0]

    shap.initjs()
    plt.figure(figsize=(20, 4))
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    shap.force_plot(expected_value, chosen_shap_values, chosen_df, matplotlib=True, show=False)
    plt.title(f'Force Plot - Вероятность дефолта: {chosen_proba:.2%}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nФакторы, повышающие риск:")
    shap_client = pd.DataFrame({
        'feature': feature_names,
        'shap_value': chosen_shap_values
    }).sort_values('shap_value', ascending=False)

    for _, row in shap_client.head(3).iterrows():
        print(f"  ↑ {row['feature']}: +{row['shap_value']:.4f}")

    print("\nФакторы, снижающие риск:")
    for _, row in shap_client.tail(3).iterrows():
        print(f"  ↓ {row['feature']}: {row['shap_value']:.4f}")

else:
    print(f"SHAP анализ пропущен (модель {best_model_name} не поддерживается)")

# =====================================================
# ЭТАП 4. СОХРАНЕНИЕ МОДЕЛИ И СОЗДАНИЕ API
# =====================================================

print("\n" + "="*60)
print("ЭТАП 4: СОХРАНЕНИЕ МОДЕЛИ И СОЗДАНИЕ API")
print("="*60)

# Сохранение модели
joblib.dump(best_model_obj, 'credit_scoring_model.pkl')
print("✓ Модель сохранена: credit_scoring_model.pkl")

# Сохранение метаданных
metadata = {
    'feature_names': numerical_cols + categorical_cols,
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'best_model_name': best_model_name,
    'auc_roc_score': float(results_df['AUC-ROC'].max())
}
joblib.dump(metadata, 'model_metadata.pkl')
print("✓ Метаданные сохранены: model_metadata.pkl")

# Создание API файла
api_code = '''# api.py - Credit Scoring API
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import uvicorn

# Загрузка модели
print("Loading model...")
model = joblib.load('credit_scoring_model.pkl')
metadata = joblib.load('model_metadata.pkl')
print(f"Model loaded: {metadata['best_model_name']}")
print(f"AUC-ROC: {metadata['auc_roc_score']:.4f}")

app = FastAPI(
    title="Credit Scoring API",
    description="Credit Default Prediction Service",
    version="1.0.0"
)

class CreditRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Client features")

class CreditResponse(BaseModel):
    prediction: int = Field(..., description="0=Approve, 1=Reject")
    probability_default: float = Field(..., description="Default probability")
    credit_decision: str = Field(..., description="Credit decision")
    risk_level: str = Field(..., description="Risk level")

@app.get("/")
def root():
    return {
        "message": "Credit Scoring API",
        "model": metadata['best_model_name'],
        "auc_roc": metadata['auc_roc_score']
    }

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

print("\n" + "="*60)
print("РАБОТА УСПЕШНО ЗАВЕРШЕНА!")
print("="*60)
print("\nИНСТРУКЦИЯ ПО ЗАПУСКУ API:")
print("1. pip install fastapi uvicorn")
print("2. python api.py")
print("3. Открыть http://127.0.0.1:8000/docs")
print("="*60)
print("   или")
print("   uvicorn api:app --reload")
print("\n3. Откройте в браузере:")
print("   http://127.0.0.1:8000/docs")
print("\n4. Используйте Swagger UI для тестирования эндпоинта /predict")
print("\n5. Пример тестового запроса сохранен в 'test_request.json'")
print("\n6. Для остановки сервера нажмите Ctrl+C")
print("=" * 60)
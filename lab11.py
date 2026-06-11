# =====================================================
# ЛАБОРАТОРНАЯ РАБОТА №11
# Тема: Сравнительный анализ архитектур для бинарной классификации
# Датасет: German Credit Data (sklearn)
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

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

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

# 1.1 Загрузка датасета German Credit Data
print("Загрузка данных...")
german_credit = fetch_openml(name='german', version=1, as_frame=True)
df = german_credit.frame
print("Данные успешно загружены!\n")

# 1.2 Первые строки, информация, описательная статистика
print("Первые 5 строк данных:")
print(df.head(), "\n")

print("Информация о данных:")
print(df.info(), "\n")

print("Описательная статистика:")
print(df.describe(), "\n")

# 1.3 Проверка пропусков
print("Пропуски в каждом столбце:")
print(df.isnull().sum(), "\n")

# Пропусков нет, обработка не требуется

# 1.4 Определение числовых и категориальных признаков
target = 'class'  # целевая переменная

# Определяем типы признаков (на основе данных)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove(target)

categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()

print(f"Числовые признаки: {numerical_cols}")
print(f"Категориальные признаки: {categorical_cols}\n")

# 1.5 Визуализация целевой переменной (баланс классов)
plt.figure()
df[target].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Распределение целевой переменной (0 - хороший, 1 - плохой)')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.xticks([0, 1], ['Хороший (0)', 'Плохой (1)'], rotation=0)
plt.grid(axis='y')
plt.show()

# Оценка баланса
class_counts = df[target].value_counts(normalize=True)
print(f"Баланс классов:\nХорошие: {class_counts[0]:.2%}\nПлохие: {class_counts[1]:.2%}\n")

# 1.6 Корреляционная матрица для числовых признаков
plt.figure(figsize=(12, 8))
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Корреляционная матрица числовых признаков')
plt.tight_layout()
plt.show()

# =====================================================
# ЭТАП 2. ПОСТРОЕНИЕ МОДЕЛЕЙ И СРАВНИТЕЛЬНЫЙ АНАЛИЗ
# =====================================================

# 2.1 Разделение на обучающую (80%) и тестовую (20%)
X = df.drop(columns=[target])
y = df[target].astype(int)  # убедимся, что целевая - int

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}\n")

# 2.2 Создание конвейера предобработки
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# 2.3 Определение моделей и параметров для GridSearchCV
models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l2']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5, 10]
        }
    },
    'LGBM': {
        'model': LGBMClassifier(random_state=42, verbose=-1),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5, 7],
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
    print(f"\n=== Обучаем {name} ===")
    
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
    
    # Предсказания на тесте
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Метрики
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'AUC-ROC': roc_auc,
        'Best Params': grid_search.best_params_,
        'Model Object': best_model
    })
    
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, AUC-ROC: {roc_auc:.4f}")

# 2.6 Сравнительная таблица результатов
results_df = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']]
print("\n" + "="*60)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*60)
print(results_df.to_string(index=False))
print("="*60)

# Визуализация сравнения моделей по AUC-ROC
plt.figure()
plt.bar(results_df['Model'], results_df['AUC-ROC'], color='skyblue')
plt.ylim(0, 1)
plt.title('Сравнение моделей по метрике AUC-ROC')
plt.ylabel('AUC-ROC')
plt.xlabel('Модель')
for i, v in enumerate(results_df['AUC-ROC']):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
plt.show()

# =====================================================
# ЭТАП 3. ИНТЕРПРЕТАЦИЯ МОДЕЛИ И АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
# =====================================================

# 3.1 Выбираем лучшую модель по AUC-ROC
best_model_name = results_df.loc[results_df['AUC-ROC'].idxmax(), 'Model']
best_model_obj = [res['Model Object'] for res in results if res['Model'] == best_model_name][0]
print(f"\nЛучшая модель по AUC-ROC: {best_model_name}")

# 3.2 SHAP анализ
# Подготавливаем данные для SHAP (предобработанные)
X_train_processed = best_model_obj.named_steps['preprocessor'].transform(X_train)
X_test_processed = best_model_obj.named_steps['preprocessor'].transform(X_test)

# Получаем названия признаков после one-hot кодирования
feature_names = (numerical_cols + 
                 list(best_model_obj.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(categorical_cols)))
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

# Для SHAP используем TreeExplainer (для RandomForest, LGBM, CatBoost)
# Если лучшая модель - LogisticRegression, используем LinearExplainer
explainer = None
if best_model_name in ['RandomForest', 'LGBM', 'CatBoost']:
    explainer = shap.TreeExplainer(best_model_obj.named_steps['classifier'])
    shap_values = explainer.shap_values(X_test_df)[1]  # для бинарной классификации берём класс 1
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title(f"SHAP важность признаков - {best_model_name}")
    plt.tight_layout()
    plt.show()
    
    # Топ-5 признаков
    shap_importance = pd.DataFrame({
        'feature': X_test_df.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    print("\nТоп-5 наиболее важных признаков (по SHAP):")
    print(shap_importance.head(5))
else:
    # Для логистической регрессии
    explainer = shap.LinearExplainer(best_model_obj.named_steps['classifier'], X_train_df)
    shap_values = explainer.shap_values(X_test_df)
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title(f"SHAP важность признаков - {best_model_name}")
    plt.tight_layout()
    plt.show()
    
    shap_importance = pd.DataFrame({
        'feature': X_test_df.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    print("\nТоп-5 наиболее важных признаков (по SHAP):")
    print(shap_importance.head(5))

# 3.4 Выбираем клиента из тестовой выборки, которому модель отказала (предсказание близко к 1)
y_pred_proba_best = best_model_obj.predict_proba(X_test)[:, 1]
# Найдём индекс клиента с предсказанием > 0.7 (плохой)
bad_idx = np.where(y_pred_proba_best > 0.7)[0]
if len(bad_idx) > 0:
    chosen_idx = bad_idx[0]
else:
    chosen_idx = np.argmax(y_pred_proba_best)  # если нет >0.7, берём максимальный

chosen_client = X_test.iloc[[chosen_idx]]
chosen_proba = y_pred_proba_best[chosen_idx]
chosen_true = y_test.iloc[chosen_idx]

print(f"\nВыбран клиент (индекс {chosen_idx})")
print(f"Истинный класс: {'Плохой' if chosen_true==1 else 'Хороший'}")
print(f"Предсказанная вероятность дефолта: {chosen_proba:.4f}")

# 3.5 Force Plot для выбранного клиента (только для Tree-based моделей)
if best_model_name in ['RandomForest', 'LGBM', 'CatBoost']:
    # Для force plot нужно передать shap_values для конкретного клиента
    shap_values_single = explainer.shap_values(chosen_client)
    # Преобразуем chosen_client через препроцессор
    chosen_processed = best_model_obj.named_steps['preprocessor'].transform(chosen_client)
    chosen_df = pd.DataFrame(chosen_processed, columns=feature_names)
    
    # force plot
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values_single[1][0], chosen_df, matplotlib=True, show=False)
    plt.title(f"Force Plot для выбранного клиента (вероятность дефолта = {chosen_proba:.2f})")
    plt.tight_layout()
    plt.show()
    
    print("\nИнтерпретация Force Plot:")
    print("Красные факторы увеличивают риск дефолта, синие - уменьшают.")
    print("Основные факторы, повлиявшие на отказ:")
    # Получаем топ-3 фактора, увеличивающие риск
    shap_client = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values_single[1][0]
    }).sort_values('shap_value', ascending=False)
    print(shap_client.head(3))
else:
    print("Для логистической регрессии force plot не поддерживается в данной версии, используйте summary plot.")

# =====================================================
# ЭТАП 4. ДЕПЛОЙ МОДЕЛИ (БОНУСНЫЙ ЭТАП)
# =====================================================

# 4.1 Сохраняем лучшую модель и препроцессор
joblib.dump(best_model_obj, 'credit_scoring_model.pkl')
print("\nМодель сохранена как 'credit_scoring_model.pkl'")

# Сохраняем также список признаков для API
feature_names_all = numerical_cols + categorical_cols
joblib.dump(feature_names_all, 'feature_names.pkl')
print("Список признаков сохранён")

# 4.2 Создаем файл api.py (будет сохранён отдельно)
api_code = """
# api.py - Веб-сервис для кредитного скоринга
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Загрузка модели и препроцессора
model = joblib.load('credit_scoring_model.pkl')
feature_names = joblib.load('feature_names.pkl')

app = FastAPI(title="Credit Scoring API", 
              description="API для предсказания вероятности дефолта по кредиту",
              version="1.0")

class ClientData(BaseModel):
    data: Dict[str, Any]

@app.get("/")
def root():
    return {"message": "Credit Scoring API is running. Use /predict endpoint."}

@app.post("/predict")
def predict(client: ClientData):
    try:
        # Преобразуем входные данные в DataFrame
        input_df = pd.DataFrame([client.data])
        
        # Проверяем, что все признаки на месте
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            raise HTTPException(status_code=400, 
                                detail=f"Missing columns: {missing_cols}")
        
        # Предсказание вероятности
        proba = model.predict_proba(input_df)[0, 1]
        prediction = 1 if proba >= 0.5 else 0
        
        return {
            "prediction": int(prediction),
            "probability_default": float(proba),
            "credit_decision": "Rejected" if prediction == 1 else "Approved"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск: uvicorn api:app --reload
"""

with open('api.py', 'w', encoding='utf-8') as f:
    f.write(api_code)
print("\nФайл api.py создан")

print("\n" + "="*60)
print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
print("="*60)
print("\nИНСТРУКЦИЯ ПО ЗАПУСКУ API-СЕРВИСА:")
print("1. Установите дополнительные библиотеки: pip install fastapi uvicorn")
print("2. Запустите сервис командой: uvicorn api:app --reload")
print("3. Откройте в браузере: http://127.0.0.1:8000/docs")
print("4. Используйте Swagger UI для тестирования эндпоинта /predict")
print("\nПример тела запроса (JSON):")
print("""
{
  "data": {
    "duration": 6,
    "amount": 1169,
    "age": 67,
    "checking_status": "no checking",
    "credit_history": "critical",
    "purpose": "radio/tv",
    "savings_status": "no known savings",
    "employment": "unemployed",
    "personal_status": "male single",
    "housing": "own",
    "job": "unskilled resident"
  }
}
""")
# api.py - Credit Scoring API
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

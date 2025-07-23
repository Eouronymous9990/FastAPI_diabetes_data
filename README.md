# 🧠 Diabetes Prediction API using FastAPI

This is a simple machine learning API built with FastAPI that predicts diabetes progression using the built-in `diabetes` dataset from scikit-learn. The model is trained using a Random Forest Regressor and returns a continuous prediction value.

## 🚀 Features

- FastAPI for building the web API
- Trained using `RandomForestRegressor` from `scikit-learn`
- Accepts normalized medical features as input
- Returns a prediction of diabetes progression

## 📦 Requirements

Install the required packages:

```bash
pip install fastapi scikit-learn uvicorn pydantic numpy

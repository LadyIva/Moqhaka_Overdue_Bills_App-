---
title: Moqhaka Overdue Bills App
emoji: 📈
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---

# Moqhaka Overdue Bills Application
This Streamlit application helps to predict and manage overdue bills for Moqhaka Local Municipality.
It utilizes an XGBoost model and a scaler to process customer data and identify accounts at risk.

## Features:
- Interactive dashboard for data exploration.
- Prediction of overdue bills for individual accounts.
- Data visualizations.

## Data and Model:
- Data is loaded from `data/df_engineered_features.parquet`.
- Model and scaler artifacts are loaded from `model_artifacts/best_xgboost_model.pkl` and `model_artifacts/scaler.joblib`.

## How to Use:
Visit the app page on Hugging Face Spaces and interact with the dashboard.
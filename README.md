# Tourism Demand Forecasting — Sri Lanka
**Student ID:** 258693L

## Overview
Machine learning-driven tourism demand forecasting system for Sri Lanka,
integrating macroeconomic indicators, Google Trends digital demand signals,
and a novel disruption severity index to predict monthly tourist arrivals.

## Models Evaluated
- Ridge Regression (baseline): RMSE 47,944 | MAPE 21.46%
- Random Forest (tuned): RMSE 40,166 | MAPE 16.11%
- LSTM Neural Network: RMSE 89,510 | MAPE 41.64%

## Key Finding
Google Trends is the dominant demand predictor (SHAP rank #1),
outperforming lagged arrival variables and confirming that digital
search intent leads actual arrivals by 1-2 months.

## Live Demo
[Streamlit App](https://your-app-url.streamlit.app)

## Project Structure
- `data/` — master dataset and preprocessed features
- `models/` — trained model pipelines (pkl)
- `dashboard_data/` — monthly CSVs for Power BI
- `graphs/` — all EDA and model evaluation plots
- `notebooks/` — Colab notebooks
- `app.py` — Streamlit web application

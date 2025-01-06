# New York City Taxi Fare Prediction

This project trains a machine learning model to predict taxi fares in New York City using a dataset from Kaggle. The dataset includes detailed information about taxi rides, such as pickup and dropoff locations, date and time, passenger count, and fare amounts.

## Dataset

The dataset is sourced from the Kaggle competition: [New York City Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction). It contains:

- **Training Data:** Historical ride data with associated fare amounts (~5 GB).
- **Test Data:** Ride details for which the fare needs to be predicted.

## Project Features

1. **Data Preprocessing**:
   - Sampled 10% of the training data to reduce runtime.
   - Addressed missing values and outliers.
   - Engineered features like trip distance, pickup/dropoff landmarks, and datetime components.

2. **Exploratory Data Analysis (EDA)**:
   - Identified data distributions, ranges, and outliers.
   - Observed that latitude and longitude values had some errors in the dataset.

3. **Model Development**:
   - Implemented baseline models (e.g., Mean Regressor).
   - Experimented with multiple algorithms:
     - Linear Regression
     - Ridge Regression
     - Random Forest
     - Gradient Boosting (XGBoost)
   - Focused on hyperparameter tuning for XGBoost for optimal performance.

4. **Evaluation Metrics**:
   - Used Root Mean Squared Error (RMSE) as the primary evaluation metric.
   - Achieved an RMSE significantly better than the baseline model.

## Installation

To run the project, you'll need the following Python libraries installed:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib opendatasets

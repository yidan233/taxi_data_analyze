# New York City Taxi Fare Prediction

This project trains a machine learning model to predict taxi fares in New York City using a dataset from Kaggle. This project is a learning project, the competition already ended.

## Dataset

The dataset is sourced from the Kaggle competition: [New York City Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction). It contains:
- Comprehensive training and test data involving location and fare information.
- Date and time of taxi trips.

## Project Features

### Data Preprocessing:
- Sampled 10% of the training data to reduce runtime.
- Addressed missing values and outliers.
- Engineered features like trip distance, pickup/dropoff landmarks, and datetime components.

### Exploratory Data Analysis (EDA):
- Identified data distributions, ranges, and outliers.
- Observed that latitude and longitude values had some errors in the dataset.

### Model Development:
- Implemented baseline models (e.g., Mean Regressor).
- Experimented with multiple algorithms:
  - Linear Regression
  - Ridge Regression
  - Lasso
  - Elastic Net
- Compared model performances based on RMSE and selected the best-performing model for final tuning.

### Hyperparameter Tuning:
- Utilized grid search and manual tuning to optimize model parameters.

## Installation

Ensure Python and pip are installed. Install the required libraries using:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib opendatasets
```


## Usage

1. **Download and load the data**:
   Use the `opendatasets` library to download data directly from Kaggle.

2. **Run preprocessing and feature engineering scripts**:
   Prepare the data by cleaning and creating new features necessary for the models.

3. **Model training and evaluation**:
   Train various models and evaluate them to select the best one.

4. **Hyperparameter tuning**:
   Fine-tune the chosen model to improve accuracy.

5. **Predict and generate submission file**:
   Predict taxi fares for the test dataset and generate a submission file for Kaggle.

## Contributing

Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

For support or queries, reach out to [your-email@example.com].

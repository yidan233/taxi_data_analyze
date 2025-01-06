# this project will train a machine learing model to predict the fare for a txi ride in New York Xity 
# dataset is taken from kaggle https://www.kaggle.com/c/new-york-city-taxi-fare-prediction
# ---------------- import/settings ----------------------
# pip install pandas numpy scikit-learn xgboost
import opendatasets as od 
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
import matplotlib.pyplot as plt

# ----------------- load data -------------------------------------------------------------
dataset_url = "https://www.kaggle.com/c/new-york-city-taxi-fare-prediction"
od.download(dataset_url)
data_dir = 'new-york-city-taxi-fare-prediction'

# ======== observations ===========
# this is a supervised learningg regression prblem 
# training data is around 5 GB 
# we are predicting the 'fare_amount' column
# ==========================================

# since it is fairly large, will wor with 1% of the data to reduce runtime 
selected_cols = "fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count".split(',')
dtypes = {'fare_amount':'float32', 
          'pickup_longitude':'float32', 
          'pickup_latitude':'float32',
          'dropoff_longitude':'float32',
          'dropoff_latitude':'float32',
          'passenger_count':'uint8'}

# sample 1% of the data to reduce runtime 
sample_fraction = 0.1
random.seed(42)
def skip_row(row_idx):
  if row_idx == 0:
    return False
  return random.random() > sample_fraction

# load train set   
df = pd.read_csv(data_dir + '/train.csv',
                 parse_dates = ['pickup_datetime'],
                 dtype = dtypes,
                 usecols= selected_cols,
                 skiprows=skip_row)

# load test set
test_df = pd.read_csv(data_dir + '/test.csv', dtype = dtypes, parse_dates=['pickup_datetime'])

# -------------------------- explore data ---------------------------------------
explore = False
if explore:
  print("--------- traning data-----------------")
  df.info()
  df.describe()
  print(df['pickup_datetime'].min(),df['pickup_datetime'].max())
  print("--------- testing data-----------------")
  test_df.info()
  test_df.describe()
  print(test_df['pickup_datetime'].min(),test_df['pickup_datetime'].max())

# ============ observations ======================
# -------- training data -----------------------
# No Missing data 
# fare_amount ranges from -52 to 499
# passenger_count ranges from 0 to 28
# error in latitue & longitude -> negative 
# data ranges 2009-01-01 00:11:46+00:00 to 2015-06-30 23:59:54+00:00
# ---------- testing data ----------------
# No Missing data 
# only up to 6 passengers 
# latitue between 42 - 49
# longitude between -75 - -72
# data ranges 2009-01-01 00:11:46+00:00 to 2015-06-30 23:59:54+00:00
# ===============================================

# ------------------------- prepare data --------------------

# 20% data will be vaidation set 
train_df, val_df = train_test_split(df,test_size = 0.2)

# drop missing data
train_df = train_df.dropna()
val_df = val_df.dropna()

# extract 
input_cols = ['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
target_cols = 'fare_amount'

train_inputs = train_df[input_cols]
train_targets = train_df[target_cols]

val_inputs = val_df[input_cols]
val_targets = val_df[target_cols]

test_inputs = test_df[input_cols]

# ----------------- create hardcoded models ----------------
# evaluate 
def rmse(targets,preds):
  return root_mean_squared_error(targets,preds)

# a model that always return the mean 
class MeanRegressor:
  def fit(self,inputs,targets):
    self.mean = targets.mean()
  
  # return (# of rows)  means
  def predict(self,inputs):
    return np.full(inputs.shape[0],self.mean)

mean_model = MeanRegressor()
mean_model.fit(train_inputs,train_targets)
train_preds = mean_model.predict(train_inputs)
val_preds = mean_model.predict(val_inputs)

print(f"the rmse of the model is {rmse(train_targets,train_preds)}, {rmse(val_targets,val_preds)}")

# =========== observations ====================
# this dumb models’s rmse is around 10
# our models have to make a lower rmse than 10!!!
# ================================================

# ----------------------- submit --------------------------
def predict_and_submit(model, fname, test_inputs) :
  test_preds = model.predict(test_inputs)
  # replace the fare_amount column of the sample submission
  sub_df = pd.read_csv(data_dir + '/sample_submission.csv')
  sub_df['fare_amount'] = test_preds
  sub_df.to_csv(fname, index = None)
  return sub_df


# ======= observations ===========
# linear regression may not be working well 
# date might be an important feature 
# hyperparamter ? 

# ---------------------- feature engineering -----------------------

# 1, date 
# seperate the date to year,month,day,weekday and hour
def add_dateparts(df,col):
  df[col + '_year'] = df[col].dt.year
  df[col + '_month'] = df[col].dt.month
  df[col + '_day'] = df[col].dt.day
  df[col + '_weekday'] = df[col].dt.weekday
  df[col + '_hour'] = df[col].dt.hour
  return df

train_df = add_dateparts(train_df,'pickup_datetime')
val_df = add_dateparts(val_df,'pickup_datetime')
test_df = add_dateparts(test_df,'pickup_datetime')

# 2, position 

# caculate the great circle distance between 2 points 
def haversine_np(lon1,lat1,lon2,lat2):
  lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  
  a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
  c = 2 * np.arcsin(np.sqrt(a))
  km = 6371 * c
  return km

def add_trip_distance(df):
    df['trip_distance'] = haversine_np(df['pickup_longitude'],
                                      df['pickup_latitude'], 
                                      df['dropoff_longitude'],
                                      df['dropoff_latitude'])

add_trip_distance(train_df)
add_trip_distance(val_df)
add_trip_distance(test_df)

# 3, add popular landmarks (airports)
jfk_lonlat = -73.7781, 40.6413
lga_lonlat = -73.8740, 40.7769
ewr_lonlat = -74.1745, 40.6895
met_lonlat = -73.9632, 40.7794
wtc_lonlat = -74.0099, 40.7126

def add_landmark_dropoff_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + '_drop_distance'] = haversine_np(lon, lat, df['dropoff_longitude'], df['dropoff_latitude'])

def add_landmarks(df):
    landmarks = [
        ('jfk', jfk_lonlat),
        ('lga', lga_lonlat),
        ('ewr', ewr_lonlat),
        ('met', met_lonlat),
        ('wtc', wtc_lonlat)
    ]
    for name, lonlat in landmarks:
        add_landmark_dropoff_distance(df, name, lonlat)

add_landmarks(train_df)
add_landmarks(val_df)
add_landmarks(test_df)

# 4, clean ( remove outlier and invalid data ) 

# ==== ranges ======
# fare_amount : 1 - 500
# longitude : -75 - -72
# latitue : 40 - 42
# passenger: 0 - 6
# ======================

def remove_outliers(df):
    return df[(df['fare_amount'] > 1.0) &
              (df['fare_amount'] < 500) &
              (df['pickup_longitude'] > -75) &
              (df['pickup_longitude'] < -72) &
              (df['dropoff_longitude'] > -75) &
              (df['dropoff_longitude'] < -72) &
              (df['pickup_latitude'] > 40) &
              (df['pickup_latitude'] < 42) &
              (df['dropoff_latitude'] > 40) &
              (df['dropoff_latitude'] < 42) &
              (df['passenger_count'] > 0) &
              (df['passenger_count'] < 7)]
train_df = remove_outliers(train_df)
val_df = remove_outliers(val_df)

# ------------------------ scaling and one hot encoding --------------------
# do it later 

# --------------------------- Models --------------------------------------- 
# ridge regression/random forest/gradient boosting/lasso/dvm/knn/decision tree
input_cols = [ 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
       'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day',
       'pickup_datetime_weekday', 'pickup_datetime_hour', 'trip_distance',
       'jfk_drop_distance', 'lga_drop_distance', 'ewr_drop_distance',
       'met_drop_distance', 'wtc_drop_distance']
target_cols = 'fare_amount'
train_inputs = train_df[input_cols]
train_targets = train_df[target_cols]
val_inputs = val_df[input_cols]
val_targets = val_df[target_cols]

def evaluate(model):
    train_preds = model.predict(train_inputs)
    train_rmse = rmse(train_targets, train_preds)
    val_preds = model.predict(val_inputs)
    val_rmse = rmse(val_targets, val_preds)
    return train_rmse, val_rmse, train_preds, val_preds

def record(train_rmse,val_rmse, modelname):
  score = (train_rmse + val_rmse)/2
  dict_models[modelname] = score
   
dict_models = {}

# gradient boosting (the best model!!!)
model_xgb = XGBRegressor(max_depth = 6, objective = 'reg:squarederror', n_estimators = 200, random_state = 42, n_jobs = -1)
model_xgb.fit(train_inputs,train_targets)
train_rmse, val_rmse, train_preds, val_preds = evaluate(model_xgb)
record(train_rmse, val_rmse, 'gradient boosting')

showOtherModels = False
if showOtherModels:
  # linear regression 
  model_linear = LinearRegression()
  model_linear.fit(train_inputs,train_targets)
  train_rmse, val_rmse,train_preds, val_preds = evaluate(model_linear )
  record(train_rmse, val_rmse, 'linear')

  # ridge regression 
  model_ridge = Ridge(random_state=42,alpha=0.9)
  model_ridge.fit(train_inputs,train_targets)
  train_rmse, val_rmse,train_preds, val_preds = evaluate(model_ridge)
  record(train_rmse, val_rmse, 'ridge')


  # random forest 
  model_rdforest = RandomForestRegressor(random_state= 42, n_jobs = -1, max_depth = 10, n_estimators= 100)
  model_rdforest.fit(train_inputs,train_targets)
  train_rmse, val_rmse,train_preds, val_preds= evaluate(model_rdforest)
  record(train_rmse, val_rmse, 'random forest')



  # lasso 
  model_lasso = Lasso()
  model_lasso.fit(train_inputs,train_targets)
  train_rmse, val_rmse, train_preds, val_preds = evaluate(model_lasso)
  record(train_rmse, val_rmse, 'lasso')

  # elastic net 
  model_elastic = ElasticNet()
  model_elastic.fit(train_inputs,train_targets)
  train_rmse, val_rmse, train_preds, val_preds = evaluate(model_elastic)
  record(train_rmse, val_rmse, 'elastic_net')

  print(dict_models)
# ==== observations ============
# seems like randient boosting has the best performance !
# =========================================

# -------------------- Tune Hyperparameters ------------
# therefore, we will be focused on tunning the XGBosst model 

# first method, graphing ?
def test_params(ModelClass, **params):
    model = ModelClass(**params).fit(train_inputs, train_targets)
    train_preds = model.predict(train_inputs)
    train_rmse = root_mean_squared_error(train_targets, train_preds)
    val_preds = model.predict(val_inputs)
    val_rmse = root_mean_squared_error(val_targets, val_preds)
    return train_rmse, val_rmse

def test_param_and_plot(ModelClass, param_name, param_values, **other_params):
    train_errors, val_errors = [], []
    for value in param_values:
        params = dict(other_params)
        params[param_name] = value
        train_rmse, val_rmse = test_params(ModelClass, **params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)

    plt.figure(figsize=(10,6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(['Training', 'Validation'])
    plt.show()

best_params = {
    'random_state': 42,
    "n_jobs": -1,
    "objective": 'reg:squarederror',
    "learning_rate":0.05
}

def graph():
  test_param_and_plot(XGBRegressor, 'n_estimators', [100, 200, 400], **best_params)
  test_param_and_plot(XGBRegressor, 'max_depth', [3, 5, 7], **best_params)
  test_param_and_plot(XGBRegressor, 'learning_rate', [0.05, 0.1, 0.2], **best_params)

# the manually tuned one based on the graph 
xgb_model_final = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42,
                               n_estimators=500, max_depth=8, learning_rate=0.08, subsample=0.8)
xgb_model_final.fit(train_inputs,train_targets)
train_rmse, val_rmse,train_preds, val_preds= evaluate(xgb_model_final)
record(train_rmse, val_rmse, 'manual')

# not working well ...
def grid_search():
  # second method - grid search 
  param_grid = {
      'max_depth': [3, 5, 7, 9],
      'n_estimators': [100, 200, 300],
      'learning_rate': [0.01, 0.05, 0.1]
  }

  grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, 
                            scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
  grid_search.fit(train_inputs, train_targets)
  best_model = grid_search.best_estimator_
  best_model.fit(train_inputs,train_targets)
  train_rmse, val_rmse,train_preds, val_preds= evaluate(best_model)
  record(train_rmse, val_rmse, 'grid search')

test_inputs = test_df[input_cols]




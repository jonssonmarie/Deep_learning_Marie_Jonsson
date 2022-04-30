"""
0. MLP for regression (*)
We will continue with the dataset that we used in the lecture with predicting miles per gallons using an MLP for regression.
  a) Load the mpg dataset using seaborn. (*)
  b) Use your data analysis skills to perform EDA. (*)
  c) Find out the missing values in the dataset and use a machine learning model to fill them in (imputation). (**)
  d) Can you figure out a way to see if the values filled in are reasonable? (**)
  e) Do a train|val|test split on the data and scale it properly. Test out which scaling method to use. (*)
  f) Create an MLP with hidden layers, 1-3, and test out different amount of nodes.
     Choose the number of epochs you want to use throughout all experiments.
     Plot training losses and validation losses for different configurations. (*)
  g) Now use early stopping to tune the number of epochs. (*)
  h) Train on all training data and validation data. (*)
  i) Predict on test data and evaluate. (*)
  j) Can you create an MLP model that beats random forest for this dataset? (**)
"""

import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Flatten, Dense, InputLayer, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from scikeras.wrappers import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import keras_tuner as kt
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping


from Assets import initial_analyse, unique_df, statistics_mse_mae_rmse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
data = sns.load_dataset("mpg").drop("name", axis=1)

# initial analyse
initial_analyse.initial_analyse(data)
unique_df.unique_names(data, ["origin", "model_year", "cylinders"])

# drop origin and model_year since they do not contribute to correct horsepower
# horsepower is ok with the type of data that is available.
# HP is usually measured in several ways, fuel type/quality affect hp as well as internal measures not
# noted in the data
impute_X = data.drop(["origin", "model_year"], axis=1).values
column_name = data.drop(["origin", "model_year"], axis=1).columns

# impute using KNN
impute_knn = KNNImputer(n_neighbors=2)
data_imputed = impute_knn.fit_transform(impute_X)

data_imputed = pd.DataFrame(data_imputed).rename({0: 'mpg', 1: 'cylinders', 2: 'displacement', 3: 'horsepower',
                                                  4: 'weight', 5: 'acceleration'}, axis=1)
data_imputed["origin"] = data["origin"]
data_imputed["model_year"] = data["model_year"]

bins = pd.IntervalIndex.from_tuples([(69, 73), (73, 77), (77, 82)])
data_imputed["model_year"] = pd.cut(data_imputed["model_year"], bins=bins)

# sns.pairplot(data_imputed[['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'horsepower']],
# diag_kind='kde')
plt.show()
print(data_imputed.describe().transpose())


def one_hot_encoder(df_to_convert, sub_lst):
    """
    :param df_to_convert: DataFrame
    :param sub_lst: list
    :return: DataFrame
    """
    # https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features
    X = df_to_convert[sub_lst]  # drop='first' but if more choices than 2 'first' needed to be removed
    drop_enc = OneHotEncoder(handle_unknown='ignore').fit(X)
    encoded = drop_enc.transform(X).toarray()
    # print(drop_enc.transform(X).toarray())
    print(drop_enc.get_feature_names_out())
    return encoded


df = one_hot_encoder(data_imputed, ["origin", "model_year"])
df = pd.DataFrame(df).rename({0: 'origin_europe', 1: 'origin_japan', 2: 'origin_usa', 3: 'model_year_(69, 73]',
                              4: 'model_year_(73, 77]', 5: 'model_year_(77, 82]'}, axis=1)

data = data_imputed.join(df, how="outer").drop(["origin", "model_year"], axis=1)

X, y = data.drop("mpg", axis=1).values, data["mpg"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

scaler1 = StandardScaler()
scaler2 = MinMaxScaler()


def scale_data(train_x, test_x, scaler):
    scaled_X_train = scaler.fit_transform(train_x)
    scaled_X_test = scaler.transform(test_x)
    return scaled_X_train, scaled_X_test


scaled_X_train, scaled_X_test = scale_data(X_train, X_test, scaler1)
scaled_X_val, scaled_X_test = scale_data(X_val, X_test, scaler1)


def return_intercept_coefficients(B_0, B_1):
    print(f"Linear reg: intercept={B_0}, coefficients={B_1}")


# Multiple linear regression
model_lin_reg = LinearRegression()
model_lin_reg.fit(scaled_X_train, y_train)

intercept, coefficients = model_lin_reg.intercept_, model_lin_reg.coef_
return_intercept_coefficients(intercept, coefficients)


def plot_history(model):
    # history är en dict med ett lager history till och består av loss och val_loss
    df_loss = pd.DataFrame(model.history.history)
    # vill inte ha från 0 utan kör range 1 då 0 är inte epoch utan 1 är första epoch
    df_loss.index = range(1, len(df_loss) + 1)
    df_loss.plot(xlabel="Epochs", ylabel="MSE loss")
    plt.show()


def return_weights_bias(model):
    # vi kan få ut weights och bias från nätverket
    weights, bias = model.layers[0].get_weights()
    # .get_weights() Returns the current weights of the layer, as NumPy arrays.
    # a Dense layer returns a list of two values: the kernel matrix and the bias vector.
    print(f"MLP {model}: {bias=}, {weights=}")


def evaluate_model(model, test_x, test_y, title):
    """
    :param model: model
    :param test_x: DataFrame
    :param test_y: DataFrame
    :return: None
    """
    y_predict = model.predict(test_x)
    print(title)
    statistics_mse_mae_rmse.statistics(test_y, y_predict)


# Artificial Neural Network (ANN) - aka shallow MLP
"""def MLP_model(nodes=None, names=None):

    model = Sequential(name="MLP_model")
    model.add(InputLayer(X_train.shape[1]))

    for node, name in zip(nodes, names):
        model.add(Dense(node, name=name))

    model.compile(loss="mean_squared_error", optimizer=SGD(learning_rate=0.001))

    return model"""


"""model_1 = MLP_model(nodes=[20, 1], names=["Hidden1", "Output_layer"])
model_1.summary()

# modellen tränas här via .fit med extra godis!
model_1.fit(scaled_X_train, y_train, epochs=200, verbose=0, validation_data=(scaled_X_val, y_val))
plot_history(model_1)

# return_weights_bias(model_1)

# Prediction and evaluation
evaluate_model(model_1, scaled_X_val, y_val, "model_1")

# Early stopping
model_1.fit(scaled_X_train, y_train, epochs=20, verbose=0, validation_data=(scaled_X_val, y_val))
plot_history(model_1)


model_2 = MLP_model(nodes=[20, 20, 1], names=["Hidden1", "Hidden2", "Output_layer"])
model_2.summary()

model_2.fit(scaled_X_train, y_train, epochs=200, verbose=0, validation_data=(scaled_X_val, y_val))
plot_history(model_2)

# return_weights_bias(model_2)
evaluate_model(model_2, scaled_X_val, y_val, "model_2")

# Early stopping
model_2.fit(scaled_X_train, y_train, epochs=20, verbose=0, validation_data=(scaled_X_val, y_val))
plot_history(model_2)


model_3 = MLP_model(nodes=[12, 10, 10, 1], names=["Hidden1", "Hidden2", "Hidden3", "Output_layer"])
model_3.summary()

model_3.fit(scaled_X_train, y_train, epochs=200, verbose=0, validation_data=(scaled_X_val, y_val))
plot_history(model_3)

# return_weights_bias(model_3)
evaluate_model(model_3, scaled_X_val, y_val, "model_3")

# Early stopping
model_3.fit(scaled_X_train, y_train, epochs=20, verbose=0, validation_data=(scaled_X_val, y_val))
plot_history(model_3)"""

pipe_RFC = Pipeline([("scaler", None), ("random_forest", RandomForestRegressor())])


def cross_validation2(train_x, train_y):
    """
    :param train_x: DataFrame
    :param train_y: DataFrame
    :return: None
    """
    param_grid_RFC = {"random_forest__n_estimators": [300],
                      "random_forest__criterion": ["squared_error", "absolute_error", "poisson"],
                      "random_forest__max_features": ["auto", "sqrt", "log2"],
                      "scaler": [StandardScaler(), MinMaxScaler()]}

    # cross validation by GridSearchCV
    scoring = ["neg_mean_squared_error", "r2", "explained_variance"]
    regressor_RFC = GridSearchCV(estimator=pipe_RFC, param_grid=param_grid_RFC, cv=10, verbose=1,
                                 scoring="neg_mean_squared_error")

    grid_search_RFC = regressor_RFC.fit(train_x, train_y)

    # Print best score and best hyperparameter
    print(f"Best Score RFC: {grid_search_RFC.best_score_:.4f} using {grid_search_RFC.best_params_}")

    # evaluate model on decided hyperparameter
    evaluate_model(regressor_RFC, scaled_X_test, y_test, "random_forest")


# cross_validation2(X_train, y_train)


def model(x_training, y_training):
    """
   r2 (coefficient of determination) regression score function.
    Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
    A constant model that always predicts the expected value of y, disregarding the input features,
    would get a  score of 0.0.
    scaled_X_val, y_val
    r2:
    Best Score RFC: 0.8545 using {'random_forest__criterion': 'squared_error', 'random_forest__max_features': 'log2', 'random_forest__n_estimators': 23, 'scaler': StandardScaler()}
    random_forest
    MSE: 144.453, MAE: 10.332, RMSE: 12.019

    "neg_mean_squared_error":
    scaled_X_val, y_val
    Best Score RFC: -8.8093 using {'random_forest__criterion': 'squared_error', 'random_forest__max_features': 'log2', 'random_forest__n_estimators': 34, 'scaler': MinMaxScaler()}
    random_forest
    MSE: 129.666, MAE: 9.734, RMSE: 11.387

    neg_mean_squared_error:
    Best Score RFC: -9.2102 using {'random_forest__criterion': 'absolute_error', 'random_forest__max_features': 'log2', 'random_forest__n_estimators': 300, 'scaler': MinMaxScaler()}
    random_forest
    MSE: 91.786, MAE: 7.980, RMSE: 9.580
    """


# model(scaled_X_test, y_test)


def create_model(nlayers=None, nnodes=None, activations=None, learn_rate=None):  # dropout_rate=0.0,
    # create model
    model = Sequential()

    model.add(Dense(12, input_dim=11, activation=activations))  # InputLayer(X_train.shape[1]
    for layer in range(1, nlayers + 1):
        model.add(Dense(nnodes, activation=activations, name=f"Hidden_{layer}"))
        #model.add(Dropout(dropout_rate))
    model.add(Dense(1, name="Output_layer"))

    # Compile model
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=learn_rate),
                  metrics=['mse'])  # SGD(learning_rate = 0.001) accuracy

    return model


model = KerasRegressor(build_fn=create_model, verbose=1, epochs=None)
# print(model.summary()) AttributeError: 'KerasRegressor' object has no attribute 'summary'

pipe_MLP = Pipeline([("scaler", None), ("model", model)])


def cross_validation(train_x, train_y):
    '''
    :param train_x: DataFrame
    :param train_y: DataFrame
    :return: None
    '''
    param_grid_MLP = {"model__epochs": [20, 40],
                      "model__nlayers": [1, 2, 3],
                      "model__nnodes": [50, 70, 80],
                      "model__learn_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
                      "model__activations": ['relu', 'sigmoid', 'linear'],

                      "scaler": [StandardScaler(), MinMaxScaler()]}
    #"model__dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],

    # cross validation by GridSearchCV
    regressor_MLP = GridSearchCV(estimator=pipe_MLP, param_grid=param_grid_MLP, cv=5, verbose=1,
                                 scoring="neg_mean_squared_error", error_score="raise")
    # fit
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=1)
    """
    Often, the first sign of no further improvement may not be the best time to stop training. 
    The model may coast into a plateau of no improvement or even get slightly worse before getting much better. 
    We can account for this by adding a delay to the trigger in terms of the number 
    of epochs on which we would like to see no improvement. This can be done by setting the “patience” argument.
    """
    grid_search_MLP = regressor_MLP.fit(train_x, train_y, model__validation_data=(X_val, y_val), model__callbacks=es)

    # Print best score and best hyperparameter
    print(f"Best Score RFC: {grid_search_MLP.best_score_:.4f} using {grid_search_MLP.best_params_}")
    """means = grid_search_MLP.cv_results_['mean_test_score']
    stds = grid_search_MLP.cv_results_['std_test_score']
    params = grid_search_MLP.cv_results_['params']"""
    """for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))"""

    # evaluate model on decided hyperparameter
    evaluate_model(regressor_MLP, scaled_X_val, y_val, "regressor_MLP")


#cross_validation(scaled_X_train, y_train)


def plot_metrics(metrics):
    _, ax = plt.subplots(1, 2, figsize=(12, 4))

    metrics[["loss", "val_loss"]].plot(ax=ax[0], title="Loss", grid=True)
    plt.show()


# Artificial Neural Network (ANN) - aka shallow MLP
def MLP_model():
    model_mlp = Sequential(name="MLP_model")
    model_mlp.add(InputLayer(X_train.shape[1]))
    model_mlp.add(Dense(50, activation="relu", name="Hidden1"))
    model_mlp.add(Dense(50, activation="relu", name="Hidden2"))

    model_mlp.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.01))

    # fit model
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, min_delta=1)
    model_mlp.fit(scaled_X_train, y_train, epochs=40, verbose=1, validation_data=(scaled_X_test, y_test), callbacks=es)
    #evaluate_model(model_mlp, scaled_X_test, y_test, "MLP_model")
    model_mlp.summary()
    MLP_history = pd.DataFrame(model_mlp.history.history)
    return MLP_history


MLP_history = MLP_model()
plot_metrics(MLP_history)



"""
med scalar 1
Best Score RFC: -9.3060 
using {
'model__activations': 'relu', 
'model__epochs': 40, 
'model__learn_rate': 0.01, 
'model__nlayers': 2, 
'model__nnodes': 50, 
'scaler': StandardScaler()}
MSE: 10.781, MAE: 2.640, RMSE: 3.283

resten försvann då det var för mycket text
1, 'model__nlayers': 2, 'model__nnodes': 50, 'scaler': MinMaxScaler()}
MSE: 5.639, MAE: 1.818, RMSE: 2.375

metrics = [["loss", "test_loss"], ["mse", "test_mse"]] ändrat test till val
    raise KeyError(f"{not_found} not in index")
KeyError: "['test_loss'] not in index"

Process finished with exit code 1


X_train mfl för evaluate_model är gjort av scalar2
param_grid_MLP = {"model__epochs": [20, 40],
                      "model__nlayers": [1, 2, 3],
                      "model__nnodes": [50, 70, 80],
                      "model__learn_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
                      "model__activations": ['relu', 'sigmoid', 'linear'],
                      "model__dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                      "scaler": [StandardScaler(), MinMaxScaler()]}


  param_grid_MLP = {"model__epochs": [20, 40],
                      "model__nlayers": [1, 2, 3],
                      "model__nnodes": [50, 70, 80],
                      "model__learn_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
                      "model__activations": ['relu', 'sigmoid', 'linear'],

                      "scaler": [StandardScaler(), MinMaxScaler()]}
                      
 Best Score RFC: -9.5821 
 using {
 'model__activations': 'relu', 
 'model__epochs': 40, 
 'model__learn_rate': 0.01, 
 'model__nlayers': 2, 
 'model__nnodes': 80, 
 'scaler': MinMaxScaler()}
 
  MSE: 14.012, MAE: 3.027, RMSE: 3.743  
 
 
 {"model__epochs": [20, 40, 60, 80, 100],
 "model__nlayers": [1, 2, 3],
 "model__nnodes": [40, 60, 80],
  "model__learn_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
 "model__activations": ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
"model__dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
 "scaler": [StandardScaler(), MinMaxScaler()]}
                      
  
Best Score RFC: -8.9049 
using {
'model__activations': 'tanh', 
'model__epochs': 80, 
'model__learn_rate': 0.01, 
'model__nlayers': 1, 
'model__nnodes': 70, 
'scaler': StandardScaler()}

MSE: 134.828, MAE: 9.959, RMSE: 11.612

Anna-Maria:  Jag fick MAE: 1.62, RMSE: 2.39 för testdatan
"""

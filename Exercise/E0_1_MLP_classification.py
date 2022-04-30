import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Flatten, Dense, InputLayer, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping

cardio_train = "../Data/data_cardio.csv"
data= pd.read_csv(cardio_train, delimiter=';')


def split_to_y_x(df, col):
    """
    :param df: DataFrame
    :param col: str
    :return: DataFrame, DataFrame
    """

    x, y = df.drop([col], axis=1), df[col]
    return x, y


x, y = split_to_y_x(data, "cardio")


def split_train_test(x_value, y_value, t_size):
    """
    :param x_value: DataFrame
    :param y_value: DataFrame
    :param t_size: float
    :return: DataFrame, DataFrame, DataFrame, DataFrame
    """
    x_train, x_test, y_train, y_test = train_test_split(x_value, y_value, test_size=t_size, random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test


# train, val, test - 80, 10, 10
X_train, X_test, y_train, y_test = split_train_test(np.array(x), np.array(y), 0.20)
X_val, X_test, y_val, y_test = split_train_test(X_test, y_test, 0.50)


def evaluate_model(model, test_x, test_y, title):
    """
    :param model: model
    :param test_x: DataFrame
    :param test_y: DataFrame
    :return: None
    """
    y_predict = model.predict(test_x)
    print(title)
    print(classification_report(test_y, y_predict))
    cm = confusion_matrix(test_y, y_predict)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()


def create_model(nlayers=None, nnodes=None, activations=None, learn_rate=None):
    # create model
    model = Sequential()

    model.add(InputLayer(X_train.shape[1]))  #  Dense(12, input_dim=11, activation=activations)
    for layer in range(1, nlayers + 1):
        model.add(Dense(nnodes, activation=activations, name=f"Hidden_{layer}"))
    model.add(Dense(1, name="Output_layer"))

    # Compile model
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learn_rate),
                  metrics=['accuracy'])  # SGD(learning_rate = 0.001)
    return model


model = KerasClassifier(build_fn=create_model, verbose=1, epochs=None)

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
                      "model__activations": ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],

                      "scaler": [StandardScaler(), MinMaxScaler()]}

    # cross validation by GridSearchCV
    regressor_MLP = GridSearchCV(estimator=pipe_MLP, param_grid=param_grid_MLP, cv=5, verbose=1,
                                 scoring="neg_mean_squared_error", error_score="raise")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=1)
    grid_search_MLP = regressor_MLP.fit(train_x, train_y, model__validation_data=(X_val, y_val), model__callbacks=es)
    print(regressor_MLP.estimator.get_params().keys())

    # Print best score and best hyperparameter
    print(f"Best Score RFC: {grid_search_MLP.best_score_:.4f} using {grid_search_MLP.best_params_}")
    means = grid_search_MLP.cv_results_['mean_test_score']
    stds = grid_search_MLP.cv_results_['std_test_score']
    params = grid_search_MLP.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # evaluate model on decided hyperparameter
    evaluate_model(regressor_MLP, X_val, y_val, "regressor_MLP")


cross_validation(X_train, y_train)


def model(x_training, y_training):
    """
    :param x_training: DataFrame
    :param y_training: DataFrame
    :return: model
    """

#model(X_test, y_test)

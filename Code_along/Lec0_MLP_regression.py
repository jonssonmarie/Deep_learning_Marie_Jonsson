import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import SGD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_iterative_imputer  # https://www.youtube.com/watch?v=m_qKhnaYZlc
from sklearn.impute import IterativeImputer

df = sns.load_dataset("mpg").drop("name", axis=1)

print(df.head())
print(df["origin"].value_counts())
print(df.info())

print(df.query("horsepower.isna()"))
# Exercise: impute the values

df.dropna(axis=0, inplace=True)
print(df.info())

df["model_year"].value_counts().sort_index().plot(kind="bar", title="Model year")

bins = pd.IntervalIndex.from_tuples([(69, 73), (74, 77), (78, 82)])
df["model_year"] = pd.cut(df["model_year"], bins=bins)
print(df.head())

df = pd.get_dummies(df, columns=["model_year", "origin"], drop_first=True)
print(df.head())

# Train|test-split
# pick out values, i.e. numpy arrays and not DataFrame or Series
X, y = df.drop("mpg", axis = 1).values, df["mpg"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

print(scaled_X_train.shape, scaled_X_test.shape)

# Multiple linear regression
model_lin_reg = LinearRegression()
model_lin_reg.fit(scaled_X_train, y_train)

intercept, coefficients = model_lin_reg.intercept_, model_lin_reg.coef_

print(intercept, coefficients)


# Artificial Neural Network (ANN) - aka shallow MLP
model_shallow = Sequential(name="Shallow_network")
model_shallow.add(InputLayer(X_train.shape[1]))
#model_shallow.add(Dense(20, name="Hidden1"))    # note no activation function --> linear activation
model_shallow.add(Dense(1, name="Output_layer"))
model_shallow.compile(loss="mean_squared_error", optimizer=SGD(learning_rate=.01))
print(model_shallow.summary())
# Vårt första neural nätverk printas här!

# modellen tränas här via .fit med extra godis!
model_shallow.fit(scaled_X_train, y_train, epochs=50, verbose=1, validation_data=(scaled_X_test, y_test))

# history är en dict med ett lager history till och består av loss och val_loss
df_loss = pd.DataFrame(model_shallow.history.history)
print(df_loss.head())
# vill inte ha 0-4 utan kör range då 0 är inte epoch utan 1 är första epoch
df_loss.index = range(1, len(df_loss)+1)
print(df_loss.head())

df_loss.plot(xlabel="Epochs", ylabel="MSE loss")
plt.show()

# vi kan få ut weights och bias från nätverket
weights, bias = model_shallow.layers[0].get_weights()
# .get_weights() Returns the current weights of the layer, as NumPy arrays.
# a Dense layer returns a list of two values: the kernel matrix and the bias vector.

print(f"Linear reg: {intercept=}, {coefficients=}")
print(f"ANN {bias=}, {weights=}")

# Prediction and evaluation
y_pred_ANN = model_shallow.predict(scaled_X_test)
y_pred_lin_reg = model_lin_reg.predict(scaled_X_test)

print("MAE, RMSE for ANN:")
print(mean_absolute_error(y_test, y_pred_ANN), np.sqrt(mean_squared_error(y_test, y_pred_ANN)))

print("MAE, RMSE for linear regression:")
print(mean_absolute_error(y_test, y_pred_lin_reg), np.sqrt(mean_squared_error(y_test, y_pred_lin_reg)))

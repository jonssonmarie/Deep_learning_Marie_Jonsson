import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def statistics(y_testing, y_predict):
    """
    MAE - Mean Absolut error =  sum(abs(x - x_bar)) / n, sum n=1 to n-1
    MSE - Mean squared error  =  sum((x - x_bar)^2) / n, sum n=1 to n-1
    RMSE - Root mean square error =  sqrt(MSE)
    :param y_testing:
    :param y_predict:
    :return: print MSE, MAE, RMSE
    """
    MSE = mean_squared_error(y_testing, y_predict)
    MAE = mean_absolute_error(y_testing, y_predict)
    RMSE = np.sqrt(MSE)

    print(f"MSE: {MSE:.3f}, MAE: {MAE:.3f}, RMSE: {RMSE:.3f}\n")

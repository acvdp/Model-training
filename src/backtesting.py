from sktime.forecasting.model_selection import SlidingWindowSplitter, ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import mean_squared_error
from sktime.forecasting.base import ForecastingHorizon
import numpy as np
import warnings

def backtesting_sliding(modelo, y, X, fh, window_length):
    
    splitter = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=fh[-1])
    errores = []
    warnings.filterwarnings("ignore")

    modelo_name = modelo.__class__.__name__.lower()

    for train_idx, test_idx in splitter.split(y):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        y_train = y_train.dropna().astype(float)
        y_test = y_test.dropna().astype(float)

        X_train = X.iloc[train_idx] if X is not None else None
        X_test = X.iloc[test_idx] if X is not None else None

        if "arima" in modelo_name:
            fh_pred = ForecastingHorizon(np.arange(1, len(y_test)+1), is_relative=True)
            modelo.fit(y_train)
            predicciones = modelo.predict(fh=fh_pred)
        elif "ltsf" in modelo_name:
            fh_pred = ForecastingHorizon(np.arange(1, len(y_test)+1), is_relative=True)
            modelo.fit(y_train, fh=fh_pred)
            predicciones = modelo.predict(fh=fh_pred)
        else:
            fh_pred = ForecastingHorizon(y_test.index, is_relative=False)
            modelo.fit(y_train, X=X_train)
            predicciones = modelo.predict(fh=fh_pred, X=X_test)

        error = mean_squared_error(y_test, predicciones)
        errores.append(error)

    return np.mean(errores), errores  

def backtesting_expanding(modelo, y, X, fh, initial_window):
   
    splitter = ExpandingWindowSplitter(fh=fh, initial_window=initial_window, step_length=fh[-1])
    errores = []
    
    modelo_name = modelo.__class__.__name__.lower()

    for train_idx, test_idx in splitter.split(y):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        y_train = y_train.dropna().astype(float)
        y_test = y_test.dropna().astype(float)

        X_train = X.iloc[train_idx] if X is not None else None
        X_test = X.iloc[test_idx] if X is not None else None
        warnings.filterwarnings("ignore")


        if "arima" in modelo_name:
            fh_pred = ForecastingHorizon(np.arange(1, len(y_test)+1), is_relative=True)
            modelo.fit(y_train)
            predicciones = modelo.predict(fh=fh_pred)
        elif "ltsf" in modelo_name:
            fh_pred = ForecastingHorizon(np.arange(1, len(y_test)+1), is_relative=True)
            modelo.fit(y_train, fh=fh_pred)
            predicciones = modelo.predict(fh=fh_pred)
        else:
            fh_pred = ForecastingHorizon(y_test.index, is_relative=False)
            modelo.fit(y_train, X=X_train)
            predicciones = modelo.predict(fh=fh_pred, X=X_test)
       
        error = mean_squared_error(y_test, predicciones)
        errores.append(error)

    return np.mean(errores), errores  

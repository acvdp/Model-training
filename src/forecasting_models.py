from sktime.forecasting.arima import ARIMA
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.ltsf import LTSFLinearForecaster
from xgboost import XGBRegressor

def definir_modelos():
    """Define y devuelve los modelos de forecasting que vamos a usar."""

    modelos = {
        # Baseline
        "NaiveForecaster": NaiveForecaster(strategy="last"),

        # Modelo Estadístico
        "ARIMA": ARIMA(order=(2, 1, 2)),

         # Modelo Machine Learning
        "XGBRegressor": make_reduction(
            XGBRegressor(n_estimators=100, max_depth=3, random_state=42),
            strategy="recursive",
           window_length=14
        ),

        #Modelo Deep Learning
        "LTSFLinear": LTSFLinearForecaster(
            seq_len=30,
            pred_len=7,
            num_epochs=10
        )
    }

    return modelos

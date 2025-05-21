# main_pipeline.py

import pandas as pd
import numpy as np
import os
from itertools import product

from src.preprocessing import *
from src.backtesting import *
from src.forecasting_models import *
from src.grid_search import *
from src.forecasting import *
from src.visualization import *
from src.evaluation import *

import warnings
warnings.filterwarnings("ignore")

# Crear carpeta para resultados si no existe
os.makedirs('results', exist_ok=True)

# Cargar y procesar datos
df = cargar_datos('data/ventas.csv')
df = procesar_datos(df)

# Definir modelos
modelos = definir_modelos()

# Definir horizontes
horizontes = {
    "1d": 1,
    "1w": 7,
    "1m": 30
}

# Inicializar lista para guardar resultados
resultados = []
resultados_errores = []
parametros = list(product([0,1,2], [0,1], [0,1,2]))
# Definir espacio de hiperparámetros para cada modelo
param_grids = {
    "ARIMA": {
        "order": [(1,1,0), (1,1,1), (2,1,2)]
    },
      
    "XGBRegressor": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1]
    },
    "NaiveForecaster": {
        "strategy": ["last", "mean"]
    },
    "LTSFLinear": {
        "seq_len": [14, 30],
        "pred_len": [1, 7, 30],
        "num_epochs": [10]
    }

}

# Procesar cada categoría
categorias = obtener_categorias(df)

for categoria in categorias:
    print(f"\nProcesando categoría: {categoria}")

    y, X = preparar_datos_categoria(df, categoria, eliminar_outliers=True, verbose=True)

    for nombre_modelo, modelo in modelos.items():
        for nombre_horizonte, horizonte in horizontes.items():

            print(f"➔ Modelo: {nombre_modelo}, Horizonte: {nombre_horizonte}")

            try:
                fh_array = np.arange(1, horizonte + 1)
                
                # Ajustar pred_len si el modelo tiene este parámetro
                if hasattr(modelo, "pred_len"):
                    try:
                        modelo.pred_len = horizonte
                    except:
                        pass

                if hasattr(modelo, "pred_length"):
                    try:
                        modelo.pred_length = horizonte
                    except:
                        pass
                
                # Backtesting Sliding
                error_sliding_medio, errores_sliding = backtesting_sliding(modelo, y, X, fh=fh_array, window_length=100)

                # Backtesting Expanding
                error_expanding_medio, errores_expanding = backtesting_expanding(modelo, y, X, fh=fh_array, initial_window=100)
                
                resultados.append({
                    "Categoria": categoria,
                    "Modelo": nombre_modelo,
                    "Horizonte": nombre_horizonte,
                    "Backtesting": "Sliding",
                    "RMSE Backtesting Sliding": error_sliding_medio,
                    "RMSE Backtesting Expanding": error_expanding_medio
                })

                # Guardar errores individuales para análisis posterior
                for idx, error in enumerate(errores_sliding):
                    resultados_errores.append({
                        "Categoria": categoria,
                        "Modelo": nombre_modelo,
                        "Horizonte": nombre_horizonte,
                        "Backtesting": "Sliding",
                        "Split": idx,
                        "RMSE": error
                    })

                for idx, error in enumerate(errores_expanding):
                    resultados_errores.append({
                        "Categoria": categoria,
                        "Modelo": nombre_modelo,
                        "Horizonte": nombre_horizonte,
                        "Backtesting": "Expanding",
                        "Split": idx,
                        "RMSE": error
                    })

              
            except Exception as e:
                print(f"Error en Modelo {nombre_modelo} para Categoría {categoria} ({nombre_horizonte}): {e}")


# Guardar resultados
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('results/backtesting_resultados.csv', index=False)
print("\nResultados guardados en 'results/backtesting_resultados.csv'")
# Guardar errores individuales
df_errores = pd.DataFrame(resultados_errores)
df_errores.to_csv('results/backtesting_errores_splits.csv', index=False)
print("\nErrores individuales guardados en 'results/backtesting_errores_splits.csv'")
#plot_backtesting_general("results/backtesting_resultados.csv")
#plot_whiskerplot_errores("results/backtesting_errores_splits.csv")
#evaluar_mejor_backtesting("results/backtesting_resultados.csv")
plot_whiskerplot_comparativo_por_horizonte("results/backtesting_errores_splits.csv")
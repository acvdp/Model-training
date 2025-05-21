import pandas as pd
import os

def evaluar_mejor_backtesting(ruta_csv, ruta_guardado="results/mejor_backtesting_por_modelo.csv"):
    """Carga resultados de backtesting, compara Sliding vs Expanding, y guarda el mejor en un CSV."""

    # Verificamos que el archivo exista
    if not os.path.exists(ruta_csv):
        print(f" Error: No se encontró el archivo {ruta_csv}")
        return
    
    # Cargamos los resultados
    resultados = pd.read_csv(ruta_csv)
    print(f" Resultados cargados desde {ruta_csv}")
    print(resultados.head())

    #  Comparamos RMSE Sliding vs Expanding
    mejores = []

    for idx, row in resultados.iterrows():
        if row["RMSE Backtesting Sliding"] < row["RMSE Backtesting Expanding"]:
            mejor_tipo = "Sliding"
            mejor_rmse = row["RMSE Backtesting Sliding"]
        else:
            mejor_tipo = "Expanding"
            mejor_rmse = row["RMSE Backtesting Expanding"]
        
        mejores.append({
            "Categoria": row["Categoria"],
            "Modelo": row["Modelo"],
            "Horizonte": row["Horizonte"],
            "Mejor Backtesting": mejor_tipo,
            "Mejor RMSE": mejor_rmse
        })

    mejores_df = pd.DataFrame(mejores)

    # Guardar en CSV
    mejores_df.to_csv(ruta_guardado, index=False)
    print(f"\n Resultados guardados en '{ruta_guardado}'")

    #  También devolvemos el DataFrame por si quieres seguir trabajando
    return mejores_df

def encontrar_mejores_modelos(ruta_predicciones="results/predicciones_futuras.csv", ruta_salida="results/mejores_modelos.csv"):
    """Encuentra el mejor modelo para cada categoría y horizonte basado en MAPE."""

    if not os.path.exists(ruta_predicciones):
        print(f" Error: No se encontró el archivo {ruta_predicciones}")
        return

    # Cargar predicciones
    resultados = pd.read_csv(ruta_predicciones)
    print(f"\nPredicciones cargadas ({resultados.shape[0]} filas)")

    # Inicializar lista para mejores modelos
    mejores_modelos = []

    # Agrupar por Categoría y Horizonte
    grupos = resultados.groupby(["Categoria", "Horizonte"])

    for (categoria, horizonte), grupo in grupos:
        # Encontrar el modelo con menor MAPE
        mejor_fila = grupo.loc[grupo["MAPE (%)"].idxmin()]

        mejores_modelos.append({
            "Categoria": categoria,
            "Horizonte": horizonte,
            "Mejor Modelo": mejor_fila["Modelo"],
            "Mejor MAPE (%)": mejor_fila["MAPE (%)"],
            "Mejor RMSE": mejor_fila["RMSE"],
            "Mejor MAE": mejor_fila["MAE"]
        })

    # Guardar resultados
    df_mejores = pd.DataFrame(mejores_modelos)
    df_mejores.to_csv(ruta_salida, index=False)

    print(f"\n Mejores modelos guardados en '{ruta_salida}'")

    return df_mejores


#  Permitir ejecución independiente
if __name__ == "__main__":
    ruta = "./../results/backtesting_resultados.csv"  # Ruta de tu CSV
    evaluar_mejor_backtesting(ruta)

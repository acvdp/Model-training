import pandas as pd

def eliminar_outliers_iqr(y, factor=1.5, verbose=False):
    """
    Elimina outliers de una serie usando el método del rango intercuartílico (IQR).

    Parámetros:
    - y: Serie pandas.
    - factor: Cuánto multiplicas el rango IQR (1.5 es estándar).
    - verbose: Si True, muestra número de outliers eliminados.

    Retorna:
    - Serie sin outliers.
    """
    if isinstance(y, pd.Series):
        q1 = y.quantile(25/100)
        q3 = y.quantile(75/100)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        mask = (y >= lower_bound) & (y <= upper_bound)
        y_filtrado = y[mask]
        y_interpolado = y_filtrado.interpolate(method="linear")

        if verbose:
            n_outliers = (~mask).sum()
            print(f"Eliminados {n_outliers} outliers ({n_outliers/len(y)*100:.2f}%)")

        return y_interpolado
    else:
        raise ValueError("El input debe ser una pandas Series.")


def cargar_datos(ruta_archivo):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(ruta_archivo, parse_dates=['Fecha'])

def procesar_datos(df):
    """Procesa los datos: relleno de nulos, tipos de columnas."""
    df = df.copy()
    df = df.fillna(method='ffill')  # o df = df.ffill() si prefieres
    return df

def obtener_categorias(df):
    """Obtiene la lista de categorías únicas."""
    return df['Categoría'].unique()

def preparar_datos_categoria(df, categoria, eliminar_outliers=False, verbose=False):
    """Prepara y separa series y covariables para una categoría específica."""

    df_categoria = df[df['Categoría'] == categoria].copy()

    # Agrupar por fecha
    df_categoria = df_categoria.groupby('Fecha').agg({
        'Ventas': 'sum',
        'Descuento': 'mean',
        'Periodo_Festivo': 'max',
        'Dia_Semana': 'first'
    }).reset_index()

    df_categoria = df_categoria.sort_values('Fecha')
    df_categoria = df_categoria.set_index('Fecha')

    # Asegurar frecuencia diaria explícita
    df_categoria = df_categoria.asfreq("D")

    y = df_categoria['Ventas']
    X = df_categoria[['Descuento', 'Periodo_Festivo', 'Dia_Semana']]

    if eliminar_outliers:
        # Eliminar outliers
        y = eliminar_outliers_iqr(y, verbose=verbose)

        # Reindexar para mantener estructura diaria
        y = y.asfreq("D")

        # Rellenar posibles huecos
        y = y.ffill()
        X = X.reindex(y.index)
        X = X.ffill()

        # RECONSTRUIR el índice de y para forzar freq
        y.index = pd.date_range(start=y.index.min(), end=y.index.max(), freq="D")

    return y, X





import pandas as pd

#df = pd.read_csv("data/titanic.csv")
#df.head()

df = pd.read_excel("./data/dataset_aviones.xlsx")


def describe_df(df):
    """
    Genera un resumen estadístico y estructural de cada columna del DataFrame.

    Argumentos:
    df (pandas.DataFrame): DataFrame de entrada que contiene las variables a analizar.

    Retorna:
    pandas.DataFrame: DataFrame resumen con columnas del DataFrame original como columnas
    y filas que describen tipo de dato, porcentaje de valores nulos, número de valores únicos
    y porcentaje de cardinalidad.
    """

    # Crear un diccionario vacío para almacenar la información de cada columna
    diccionario = {}

    # Iterar sobre cada columna del DataFrame
    for col in df.columns:
        
        # Obtener el tipo de dato de la columna
        tipo_dato = df[col].dtype

        # Calcular el porcentaje de valores nulos
        porcentaje_nulos = df[col].isnull().mean() * 100

        # Calcular el número de valores únicos
        valores_unicos = df[col].nunique()

        # Calcular el porcentaje de cardinalidad (valores únicos respecto al total de filas)
        porcentaje_cardinalidad = (valores_unicos / len(df)) * 100

        # Guardar los resultados en el diccionario bajo el nombre de la columna
        diccionario[col] = {
            "tipo": tipo_dato,
            "porcentaje_nulos": porcentaje_nulos,
            "valores_unicos": valores_unicos,
            "porcentaje_cardinalidad": porcentaje_cardinalidad
        }

    # Convertir el diccionario en un DataFrame
    df_resumen = pd.DataFrame(diccionario)

    # Retornar el DataFrame resumen
    return df_resumen

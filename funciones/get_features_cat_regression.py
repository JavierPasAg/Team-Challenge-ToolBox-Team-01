import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, f_oneway
def get_features_cat_regression(df,target_col,pvalue = 0.05):

    '''
    Esta función recibe un dataframe y una columna del mismo, que debería ser el target de un modelo de regresión,
    por lo que debe ser debe ser una variable numérica continua o discreta

    Como resultado devolverá una lista de las variables categóricas del dataframe cuya relación con la variable target 
    supere en confianza el test correspondiente.

    Argumentos:
        df (DataFrame): Un Dataframe cualquiera cuya variable target vamos a evaluar su relación con las categóricas del mismo.
        target_col (str): Nombre de la columna numérica del Dataframe.
        p_value (float): Nivel de confianza estadística
            valor 0.05 por defecto.

    Retorna:
Una lista con los nombre de las columnas categóricas cuya confianza sea significativa.
'''

    if not (isinstance(df,pd.DataFrame) and (isinstance(target_col,str)) and (isinstance(pvalue,float))):
        if isinstance(pvalue,int):
            pvalue = float(pvalue)
        else:
            print(f"Los tipos de los argumentos de entrada arg1: {type(df)}, arg2: {type(target_col)} y arg3: {type(pvalue)}\n"
                  "no se corresponden con los tipos requeridos: arg1: pd.DataFrame, arg2: string(str) y arg3: float.")
        return None
    # Compruebo que target col sea numérica continua
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"{target_col} no es una columna numérica")
        return None
    if df[target_col].nunique() < 10:
        print(f"{target_col} No es numérica continua.")
        return None

    cols_categoricas = df.select_dtypes(exclude=[np.number]).columns.tolist()
    columnas_result = []
    for col in cols_categoricas:
        valores = df[col].dropna().unique()
        # Si la cantidad de valores únicos de la columna categórica es = 2, haremos la prueba U de Mann-Whitney
        if len(valores) == 2:
            grupo_a = df.loc[df[col] == valores[0], target_col].dropna()
            grupo_b = df.loc[df[col] == valores[1], target_col].dropna()
            u_stat, p_valor = mannwhitneyu(grupo_a, grupo_b)
            if p_valor <= pvalue:
                columnas_result.append(col)
        # Si es distinta haremos U con ANOVA
        else:
            lista_valores = [df[df[col] == valor][target_col] for valor in valores] 
            f_val, p_valor = f_oneway(*lista_valores) 

            if p_valor <= pvalue:
                columnas_result.append(col)
    return columnas_result
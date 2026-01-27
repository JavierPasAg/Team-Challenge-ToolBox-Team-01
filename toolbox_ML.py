# Ejecutable con las funciones implementadas
import pandas as pd
import numpy as np
from scipy import stats
from funciones.tipifica_variables import tipifica_variables
from funciones.get_features_cat_regression import get_features_cat_regression

df_titanic = pd.read_csv("./data/titanic.csv")
df_aviones = pd.read_excel("./data/dataset_aviones.xlsx")

prueba_tipifica_1 = tipifica_variables(df_titanic,2,20,show_cardinalidad=True)
print(prueba_tipifica_1)

prueba_tipifica_2 = tipifica_variables(df_aviones,2,20,show_cardinalidad=True)
print(prueba_tipifica_2)

prueba_cat_regression_1 = get_features_cat_regression(df_titanic,"age")
print(prueba_cat_regression_1)

prueba_cat_regression_2 = get_features_cat_regression(df_aviones,"consumo_kg")
print(prueba_cat_regression_2)


def get_features_num_regression(df, target_col, umbral_corr=0, pvalue=None):
    
    # Verificamos que df sea un DataFrame de pandas. Si nos pasan otra cosa (lista, string, etc.) el código fallaría
    if not isinstance(df, pd.DataFrame):
        print("Error: El argumento 'df' debe ser un DataFrame de pandas.")
        return None
    
    # Verificamos que el dataframe no esté vacío
    if df.empty:
        print("Error: El DataFrame está vacío.")
        return None
 
    # Verificamos que target_col sea un string. El nombre de una columna siempre es un string
    if not isinstance(target_col, str):
        print("Error: El argumento 'target_col' debe ser un string.")
        return None
    
    # Verificamos que target_col exista en el dataframe. Si no existe, no podemos calcular correlaciones con ella
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no existe en el DataFrame.")
        print(f"Columnas disponibles: {list(df.columns)}")
        return None
    
    # Verificamos que target_col sea numérica. La correlación de Pearson solo funciona con variables numéricas
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: La columna '{target_col}' no es numérica.")
        print(f"Tipo actual: {df[target_col].dtype}")
        return None
    
    # Verificamos que tenga alta cardinalidad (que no sea binaria)
    cardinalidad_target = df[target_col].nunique()
    if cardinalidad_target <= 2:
        print(f"Error: La columna '{target_col}' tiene cardinalidad {cardinalidad_target}.")
        print("Para regresión se necesita una variable numérica con alta cardinalidad.")
        return None
   
    # Verificamos que umbral_corr sea un número. No podemos comparar correlaciones con un string
    if not isinstance(umbral_corr, (int, float)):
        print("Error: El argumento 'umbral_corr' debe ser un número.")
        return None
    
    # Verificamos que esté entre 0 y 1. 
    if not (0 <= umbral_corr <= 1):
        print("Error: El argumento 'umbral_corr' debe estar entre 0 y 1.")
        print(f"Valor recibido: {umbral_corr}")
        return None
    
   
    # pvalue puede ser None (no se aplica test) o un número entre 0 y 1
    if pvalue is not None:
        # Si no es None, verificamos que sea un número
        if not isinstance(pvalue, (int, float)):
            print("Error: El argumento 'pvalue' debe ser None o un número entre 0 y 1.")
            return None
        
        # Verificamos que esté entre 0 y 11
        if not (0 < pvalue <= 1):
            print("Error: El argumento 'pvalue' debe estar entre 0 y 1 (exclusivo de 0).")
            print(f"Valor recibido: {pvalue}")
            return None
    
    # Seleccionamos solo las columnas que son numéricas 
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Quitamos el target de la lista (no queremos correlación consigo misma)
    if target_col in columnas_numericas:
        columnas_numericas.remove(target_col)
    
    # Si no hay columnas numéricas para analizar, devolvemos lista vacía
    if len(columnas_numericas) == 0:
        print("Advertencia: No hay columnas numéricas en el DataFrame (aparte del target).")
        return []
    
    columnas_seleccionadas = []
    
    for col in columnas_numericas:
        # Obtenemos los datos sin valores nulos
        datos_validos = df[[col, target_col]].dropna()
        
        # Si no hay suficientes datos, saltamos esta columna.Necesitamos al menos 3 pares de datos para calcular correlación
        if len(datos_validos) < 3:
            print(f"Advertencia: La columna '{col}' tiene menos de 3 valores válidos. Se omite.")
            continue
        
        # Calculamos correlación de Pearson y su p-valor
        correlacion, p_valor_test = stats.pearsonr(
            datos_validos[col], 
            datos_validos[target_col]
        )
        
        # Filtramos por ubral de correlación
        if abs(correlacion) > umbral_corr:
            
            if pvalue is not None:
                if p_valor_test < pvalue:
                    # La correlación es significativa, añadimos la columna
                    columnas_seleccionadas.append(col)
                # Si no es significativa, no la añadimos (no hacemos nada)
            else:
                # No se pidió filtrar por pvalue, añadimos directamente
                columnas_seleccionadas.append(col) 
    return columnas_seleccionadas


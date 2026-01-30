# Ejecutable con las funciones implementadas
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from funciones import describe_df,tipifica_variables,get_features_num_regression,get_features_cat_regression

df_titanic = pd.read_csv("./data/titanic.csv")
df_aviones = pd.read_excel("./data/dataset_aviones.xlsx")
sep_string = "------------------------------------------"

# FUNCIONES
# describe_df
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

# tipifica_variables
def tipifica_variables(df,umbral_categoria,umbral_continua, show_cardinalidad = False):
    '''
    Esta función calcula la cardinalidad del dataframe recibido como parámetro 
    y clasifica cada columna del mismo como una variable a la que le asigna una sugerencia del tipo de variable
    que considera en función de los umbrales indicados.
    Esta clasificación la devuelve en forma de un nuevo dataframe.
    También puede mostrar la cardinalidad como una columna extra si se especifica.

    Argumentos:
        df (DataFrame): Un Dataframe cualquiera del cuál se quiera obtener la cardinalidad de sus columnas y la sugerencia de tipos.
        umbral_categoria (int): Un entero cuyo valor varía en función del dataframe recibido y que determina si una variable es categórica o no.
        umbral_continua (float): Un float que representa un porcentaje que categoriza las variables en numéricas continuas o discretas.
        show_cardinalidad (bool): Un valor True o False que establece si se quiere mostrar en el dataframe resultante la cardinalidad o no.
            valor False por defecto.

    Retorna:
    Esta función devuelve un dataframe con las columnas del recibido como parámetro, establecidas como variables en una primera columna,
    su "tipo sugerido" con el resultado de la clasificación y una tercera columna opcional "cardinalidad" que mostraría 
    la cardinalidad de cada variable.
    '''
    columnas = []
    cardinalidades = []
    tipos = []
    for columna in df:
        cardinalidad = df[columna].nunique()/len(df) * 100
        columnas.append(columna)
        cardinalidades.append(round(cardinalidad,2))
        if cardinalidad == 2:
            tipos.append("Binaria")
        elif cardinalidad < umbral_categoria:
            tipos.append("Categórica")
        elif cardinalidad >= umbral_categoria:
            if cardinalidad >= umbral_continua:
                tipos.append("Numérica continua")
            else:
                tipos.append("Numérica discreta")
                
        if show_cardinalidad:
            df_result = pd.DataFrame({"nombre_variable": columnas, "cardinalidad": cardinalidades, "tipo_sugerido": tipos})
        else:
            df_result = pd.DataFrame({"nombre_variable": columnas, "tipo_sugerido": tipos})
        
    return df_result


# get_features_num_regression
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

# plot_features_num_regression


# get_features_cat_regression
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

# plot_features_cat_regression
def plot_features_cat_regression(df, feature, target):
    """
    Visualiza la relación entre una variable categórica y un target numérico.
    
    Parámetros:
    df : DataFrame
    feature : str -> nombre de la columna categórica
    target : str -> nombre del target numérico
    """

    # Comprobaciones básicas
    if feature not in df.columns:
        raise ValueError(f"La columna '{feature}' no existe en el DataFrame.")
    if target not in df.columns:
        raise ValueError(f"La columna '{target}' no existe en el DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError("El target debe ser numérico para un análisis de regresión.")

    # Agrupación
    summary = df.groupby(feature)[target].mean().sort_values(ascending=False)

    # Gráfico
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df,
        x=feature,
        y=target,
        estimator="mean",
        errorbar="sd",
        order=summary.index,
        palette="viridis"
    )

    plt.title(f"Media de {target} por categoría de {feature}")
    plt.xlabel(feature)
    plt.ylabel(f"Media de {target}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Testeo de las funciones

print("Probamos describe_df en los dos datasets\n")
describe_titanic = describe_df(df_titanic)
print(describe_titanic,"\n")

describe_aviones = describe_df(df_aviones)
print(describe_aviones,"\n")

print(f"\n{sep_string}\n")
print("Probamos tipifica_variables en los dos datasets\n")
prueba_tipifica_1 = tipifica_variables(df_titanic,2,20,show_cardinalidad=True)
print(prueba_tipifica_1,"\n")

prueba_tipifica_2 = tipifica_variables(df_aviones,2,20,show_cardinalidad=True)
print(prueba_tipifica_2,"\n")

print(f"\n{sep_string}\n")
print("Probamos get_features_num_regression en los dos datasets\n")
prueba_get_num_regression_1 = get_features_num_regression(df_titanic,"age")
print(prueba_get_num_regression_1,"\n")

prueba_get_num_regression_2 = get_features_num_regression(df_aviones,"consumo_kg")
print(prueba_get_num_regression_2,"\n")

print(f"\n{sep_string}\n")
print("Probamos get_features_cat_regression en los dos datasets\n")
prueba_get_cat_regression_1 = get_features_cat_regression(df_titanic,"age")
print(prueba_get_cat_regression_1,"\n")

prueba_get_cat_regression_2 = get_features_cat_regression(df_aviones,"consumo_kg")
print(prueba_get_cat_regression_2,"\n")

print(f"\n{sep_string}\n")
print("Probamos plot_features_cat_regression en los dos datasets\n")
prueba_plot_cat_regression_1 = plot_features_cat_regression(df_titanic,"sex","age")
print(prueba_plot_cat_regression_1,"\n")

prueba_plot_cat_regression_2 = plot_features_cat_regression(df_aviones,"Aircompany","consumo_kg")
print(prueba_plot_cat_regression_2,"\n")


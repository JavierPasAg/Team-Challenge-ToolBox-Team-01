# Ejecutable con las funciones implementadas
import pandas as pd
import numpy as np
from scipy import stats
from funciones import describe_df,tipifica_variables,get_features_num_regression,get_features_cat_regression

df_titanic = pd.read_csv("./data/titanic.csv")
df_aviones = pd.read_excel("./data/dataset_aviones.xlsx")
sep_string = "------------------------------------------"

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
prueba_num_regression_1 = get_features_num_regression(df_titanic,"age")
print(prueba_num_regression_1,"\n")

prueba_num_regression_2 = get_features_num_regression(df_aviones,"consumo_kg")
print(prueba_num_regression_2,"\n")

print(f"\n{sep_string}\n")
print("Probamos get_features_cat_regression en los dos datasets\n")
prueba_cat_regression_1 = get_features_cat_regression(df_titanic,"age")
print(prueba_cat_regression_1,"\n")

prueba_cat_regression_2 = get_features_cat_regression(df_aviones,"consumo_kg")
print(prueba_cat_regression_2,"\n")






#Plot_featurea_cat_regresion.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
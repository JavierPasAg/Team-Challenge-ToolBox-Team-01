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
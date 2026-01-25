# Ejecutable con las funciones implementadas
import pandas as pd
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
# Ejecutable con las funciones implementadas
import pandas as pd
from funciones.tipifica_variables import tipifica_variables


df_titanic = pd.read_csv("./data/titanic.csv")
print(df_titanic.head(5))
df_result = tipifica_variables(df_titanic,2,2,show_cardinalidad=True)

print(df_result.head(5))
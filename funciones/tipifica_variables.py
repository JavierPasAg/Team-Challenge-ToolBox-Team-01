'''
Esta función debe recibir como argumento un dataframe, un entero (umbral_categoria) y un float (umbral_continua).
La función debe devolver un dataframe con dos columnas "nombre_variable", "tipo_sugerido" que tendrá tantas filas como columnas el dataframe.
En cada fila irá el nombre de una de las columnas y una sugerencia del tipo de variable.
Esta sugerencia se hará siguiendo las siguientes pautas:

Si la cardinalidad es 2, asignara "Binaria"
Si la cardinalidad es menor que umbral_categoria asignara "Categórica"
Si la cardinalidad es mayor o igual que umbral_categoria, entonces entra en juego el tercer argumento:
    Si además el porcentaje de cardinalidad es superior o igual a umbral_continua, asigna "Numerica Continua"
En caso contrario, asigna "Numerica Discreta"
'''
import pandas as pd
def tipifica_variables(df,umbral_categoria,umbral_continua, show_cardinalidad = False):
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

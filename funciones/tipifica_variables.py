import pandas as pd
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

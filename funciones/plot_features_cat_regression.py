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
import pandas as pd
import numpy as np

def optimize_memory(df):
    """
    Optimise l'utilisation de la mémoire d'un DataFrame en ajustant les types de données.
    Ignore les colonnes contenant du texte (strings/objects).
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Utilisation mémoire initiale : {start_mem:.2f} MB")
    
    for col in df.columns:
        # La ligne magique : on vérifie que la colonne est bien numérique !
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = df[col].dtypes
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Optimisation des entiers
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            # Optimisation des nombres à virgule (floats)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Utilisation mémoire après optimisation : {end_mem:.2f} MB")
    
    return df
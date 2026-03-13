import pandas as pd
import numpy as np

def optimize_memory(df):
    """
    Optimise l'utilisation de la mémoire d'un DataFrame.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = df[col].dtypes
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Mémoire : {start_mem:.2f}MB -> {end_mem:.2f}MB")
    return df

def preprocess_data(df):
    """
    LA FONCTION MANQUANTE : Elle prépare les données et utilise ton optimisation.
    """
    # 1. On nettoie les données (tu peux ajouter ta logique ici)
    df_clean = df.copy()
    
    # 2. On utilise ta fonction magique pour gagner de la place
    df_optimized = optimize_memory(df_clean)
    
    return df_optimized
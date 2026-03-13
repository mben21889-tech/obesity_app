import pytest
import pandas as pd
# On importe ta fonction depuis ton dossier src
from src.data_processing import preprocess_data 

def test_preprocess_data_is_dataframe():
    """Vérifie que la fonction renvoie bien un DataFrame"""
    # Données de test (profil type comme le tien)
    df_test = pd.DataFrame({
        'Age': [20], 
        'Height': [1.75], 
        'Weight': [54]
    })
    
    result = preprocess_data(df_test)
    
    # Le test : on vérifie que le résultat est un DataFrame
    assert isinstance(result, pd.DataFrame), "La fonction devrait renvoyer un DataFrame"

def test_preprocess_data_no_nan():
    """Vérifie qu'il n'y a pas de valeurs manquantes après le traitement"""
    df_test = pd.DataFrame({
        'Age': [20], 
        'Height': [1.75], 
        'Weight': [54]
    })
    
    result = preprocess_data(df_test)
    
    # Le test : on vérifie qu'il n'y a pas de 'NaN' (cases vides)
    assert result.isnull().sum().sum() == 0, "Il reste des valeurs nulles dans les données"

def test_preprocess_data_columns():
    """Vérifie que le traitement ne supprime pas de lignes par erreur"""
    df_test = pd.DataFrame({
        'Age': [20, 25, 30], 
        'Height': [1.75, 1.80, 1.65], 
        'Weight': [54, 80, 70]
    })
    
    result = preprocess_data(df_test)
    
    # Le test : on a envoyé 3 lignes, on doit en récupérer 3
    assert len(result) == 3, "Le nombre de lignes a changé pendant le traitement"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
df = pd.read_csv('.../data/DatasetObesity.csv')
print(df.shape)
print(df.head())


# ============================================
# 2. VALEURS MANQUANTES
# ============================================
print("Valeurs manquantes")
print(df.isnull().sum())
print(f"\nTotal : {df.isnull().sum().sum()} valeurs manquantes")

# ============================================
# 3. INFOS GÉNÉRALES
# ============================================
print("Infos générales")
df.info()
print("\nStatistiques descriptives")
print(df.describe())

# ============================================
# 4. DISTRIBUTION DES CLASSES
# ============================================
print("=== Distribution des classes ===")
print(df['NObeyesdad'].value_counts())
print("\nEn pourcentage :")
print(df['NObeyesdad'].value_counts(normalize=True).round(3) * 100)
 
plt.figure(figsize=(10, 5))
df['NObeyesdad'].value_counts().plot(kind='bar', color='steelblue')
plt.title("Distribution des niveaux d'obésité")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
 

 # ============================================
# 5. BOXPLOTS DES COLONNES NUMÉRIQUES
# ============================================
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
 
plt.figure(figsize=(14, 6))
for i, col in enumerate(numerical_cols):
    plt.subplot(2, len(numerical_cols)//2 + 1, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# ============================================
# 7. DOUBLONS
# ============================================
print(f"Nombre de doublons : {df.duplicated().sum()}")
print(df[df.duplicated()])
 
df = df.drop_duplicates()
print(f"Shape après suppression : {df.shape}")
print(f"Doublons restants : {df.duplicated().sum()}")
print("Doublons supprimés !")

# ============================================
# 8. MATRICE DE CORRÉLATION
# ============================================
plt.figure(figsize=(12, 8))
sns.heatmap(
    df.select_dtypes(include=['float64', 'int64']).corr(),
    annot=True,
    fmt='.2f',
    cmap='coolwarm'
)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()
 
# ============================================
# 9. SAUVEGARDER LES DONNÉES NETTOYÉES
# ============================================
df.to_csv('DatasetObesity_cleaned.csv', index=False)
print(f"\nFichier sauvegardé : DatasetObesity_cleaned.csv")
print(f"Dimensions finales : {df.shape}")
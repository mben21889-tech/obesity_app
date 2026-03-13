import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Ajouter le dossier src au path pour utiliser notre fonction d'optimisation
sys.path.append(os.path.abspath('src'))
from data_processing import optimize_memory

def evaluate_best_model():
    print("📊 Démarrage de l'évaluation du modèle en production...")
    
    # 1. Vérification de l'existence du modèle
    if not os.path.exists('app/best_model.pkl'):
        print("❌ Erreur : Le modèle 'best_model.pkl' est introuvable. Veuillez exécuter 'train_model.py' en premier.")
        return

    # 2. Chargement des données et des outils de preprocessing
    df = pd.read_csv('data/ObesityDataSet.csv')
    df = optimize_memory(df)
    
    model = joblib.load('app/best_model.pkl')
    le = joblib.load('app/label_encoder.pkl')
    feature_cols = joblib.load('app/feature_columns.pkl')
    
    # 3. Préparation des données 
    X = df.drop('NObeyesdad', axis=1)
    y_text = df['NObeyesdad']
    
    # Appliquer le même encodage que lors de l'entraînement
    X_encoded = pd.get_dummies(X, drop_first=True)
    for col in feature_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[feature_cols] # Forcer l'ordre exact des colonnes
    
    y = le.transform(y_text)
    
    # Recréer exactement le même jeu de test (grâce au random_state=42)
    _, X_test, _, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # 5. Calcul des métriques exigées par le projet
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    
    print("\n=============================================")
    print("🏆 RÉSULTATS DE L'ÉVALUATION (Jeu de Test) 🏆")
    print("=============================================")
    print(f"Accuracy  : {acc:.4f}  (Précision globale)")
    print(f"Precision : {prec:.4f}  (Pertinence des prédictions)")
    print(f"Recall    : {rec:.4f}  (Capacité à trouver tous les cas)")
    print(f"F1-Score  : {f1:.4f}  (Moyenne harmonique)")
    print(f"ROC-AUC   : {roc_auc:.4f}  (Capacité de discrimination)")
    print("=============================================")

    # 6. Bonus Pro : Création d'une Matrice de Confusion graphique
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matrice de Confusion du Modèle Final')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Sauvegarde de l'image
    os.makedirs('app', exist_ok=True)
    plt.savefig('app/confusion_matrix.png')
    print("\n✅ Matrice de confusion générée et sauvegardée sous 'app/confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_best_model()
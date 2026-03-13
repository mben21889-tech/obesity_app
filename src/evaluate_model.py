import matplotlib
# CRUCIAL : Force matplotlib à ne pas ouvrir de fenêtre (évite l'erreur Tkinter)
matplotlib.use('Agg') 

import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Gestion robuste des chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from data_processing import optimize_memory

def evaluate_best_model():
    print("📊 Démarrage de l'évaluation du modèle en production...")
    
    # Définition des chemins
    model_path = os.path.join(BASE_DIR, 'app', 'best_model.pkl')
    data_path = os.path.join(BASE_DIR, 'data', 'ObesityDataSet.csv')
    le_path = os.path.join(BASE_DIR, 'app', 'label_encoder.pkl')
    feat_path = os.path.join(BASE_DIR, 'app', 'feature_columns.pkl')

    # 1. Vérification de l'existence du modèle
    if not os.path.exists(model_path):
        print(f"❌ Erreur : Le modèle est introuvable dans {model_path}")
        return

    # 2. Chargement des données et des outils
    print("📂 Chargement des données et des fichiers pkl...")
    df = pd.read_csv(data_path)
    df = optimize_memory(df)
    
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    feature_cols = joblib.load(feat_path)
    
    # 3. Préparation des données 
    X = df.drop('NObeyesdad', axis=1)
    y_text = df['NObeyesdad']
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    for col in feature_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[feature_cols] 
    
    y = le.transform(y_text)
    
    # Recréer le même jeu de test
    _, X_test, _, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Prédictions
    print("🤖 Calcul des prédictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # 5. Calcul des métriques
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

    # 6. Création de la Matrice de Confusion graphique
    print("🎨 Génération de la matrice de confusion...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matrice de Confusion du Modèle Final')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Sauvegarde de l'image
    output_image = os.path.join(BASE_DIR, 'app', 'confusion_matrix.png')
    plt.savefig(output_image)
    plt.close() # Important : libère la mémoire
    print(f"\n✅ Matrice de confusion sauvegardée sous : {output_image}")

if __name__ == "__main__":
    evaluate_best_model()






import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import sys

# 1. GESTION DES CHEMINS (Pour éviter les FileNotFoundError)
# Récupère le chemin du dossier 'obesity_app' (parent de 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from data_processing import optimize_memory

# Définition des dossiers de données et de sortie
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ObesityDataSet.csv')
APP_DIR = os.path.join(BASE_DIR, 'app')
os.makedirs(APP_DIR, exist_ok=True)

# 2. CHARGEMENT ET OPTIMISATION
print(f"🔍 Recherche du fichier dans : {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"❌ ERREUR : Le fichier est introuvable !")
    print(f"Assurez-vous qu'il est bien dans : {os.path.join(BASE_DIR, 'data')}")
    sys.exit(1)

print("✅ Chargement des données...")
df = pd.read_csv(DATA_PATH)

print("⚡ Optimisation de la mémoire...")
df = optimize_memory(df)

# 3. PRÉPARATION DES DONNÉES
X = df.drop('NObeyesdad', axis=1)
y_text = df['NObeyesdad']

print("🔢 Encodage des variables...")
X_encoded = pd.get_dummies(X, drop_first=True)

le = LabelEncoder()
y = le.fit_transform(y_text)

# Sauvegarde des outils pour Streamlit
joblib.dump(le, os.path.join(APP_DIR, 'label_encoder.pkl'))
joblib.dump(list(X_encoded.columns), os.path.join(APP_DIR, 'feature_columns.pkl'))

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# 4. ENTRAÎNEMENT
models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
}

print("\n--- 🧠 Début de l'entraînement ---")
best_model_name = ""
best_f1_score = 0
best_model = None

for name, model in models.items():
    print(f"Entraînement de {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"-> {name} terminé (F1-Score: {f1:.4f})")

    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_name = name
        best_model = model

# 5. SAUVEGARDE FINALE
print(f"\n🏆 Meilleur modèle : {best_model_name} ({best_f1_score:.4f})")
MODEL_SAVE_PATH = os.path.join(APP_DIR, 'best_model.pkl')
joblib.dump(best_model, MODEL_SAVE_PATH)
print(f"✅ Succès ! Modèle enregistré dans : {MODEL_SAVE_PATH}")
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

# --- AJOUT POUR LE PROJET : Import de la fonction d'optimisation ---
sys.path.append(os.path.abspath('src'))
from data_processing import optimize_memory
# -------------------------------------------------------------------

# 1. Chargement des données (Le chemin est corrigé ici : 'data/...')
print("Chargement des données...")
df = pd.read_csv('data/ObesityDataSet.csv')

# --- AJOUT POUR LE PROJET : Optimisation de la mémoire ---
print("Optimisation de la mémoire en cours...")
df = optimize_memory(df)
# ---------------------------------------------------------

X = df.drop('NObeyesdad', axis=1)
y_text = df['NObeyesdad']

print("Encodage des variables textuelles en nombres...")
X_encoded = pd.get_dummies(X, drop_first=True)

le = LabelEncoder()
y = le.fit_transform(y_text)

os.makedirs('app', exist_ok=True)
joblib.dump(le, 'app/label_encoder.pkl')
joblib.dump(list(X_encoded.columns), 'app/feature_columns.pkl')
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
}

print("\n--- Début de l'entraînement et de l'évaluation ---")
best_model_name = ""
best_f1_score = 0
best_model = None

for name, model in models.items():
    print(f"\nEntraînement de {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

    print(f"Performances de {name} :")
    print(f"Accuracy  : {acc:.4f} | Precision : {prec:.4f} | Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f} | ROC-AUC   : {roc_auc:.4f}")

    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_name = name
        best_model = model

print(f"\n🏆 Le meilleur modèle est {best_model_name} avec un F1-Score de {best_f1_score:.4f}.")
print("Sauvegarde du modèle pour l'interface web...")

joblib.dump(best_model, 'app/best_model.pkl')
print("✅ Modèle sauvegardé sous 'app/best_model.pkl' !")
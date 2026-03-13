import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

MODEL_PATH    = r'C:\Users\HP\OneDrive - Ecole Centrale Casablanca\Bureau\obesity_app\app\best_model.pkl'
ENCODER_PATH  = r'C:\Users\HP\OneDrive - Ecole Centrale Casablanca\Bureau\obesity_app\app\label_encoder.pkl'
FEATURES_PATH = r'C:\Users\HP\OneDrive - Ecole Centrale Casablanca\Bureau\obesity_app\app\feature_columns.pkl'
DATA_PATH     = r'C:\Users\HP\OneDrive - Ecole Centrale Casablanca\Bureau\obesity_app\notebooks\DatasetObesity_cleaned_backup.csv'
OUTPUT_IMG    = r'C:\Users\HP\OneDrive - Ecole Centrale Casablanca\Bureau\obesity_app\app\shap_summary.png'

model        = joblib.load(MODEL_PATH)
le           = joblib.load(ENCODER_PATH)
feature_cols = joblib.load(FEATURES_PATH)
print('OK modele charge')

df = pd.read_csv(DATA_PATH)
if 'NObeyesdad' in df.columns:
    df = df.drop(columns=['NObeyesdad'])

df_encoded = pd.get_dummies(df)
for col in feature_cols:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
X = df_encoded[feature_cols]
print(f'X shape : {X.shape}')

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
class_names = list(le.classes_)
print(f'OK SHAP calcule')

plt.figure(figsize=(14, 8))
shap.summary_plot(shap_values, X, plot_type='bar', class_names=class_names, max_display=23, show=False)
plt.title('Importance globale des facteurs - SHAP', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=300, facecolor='white')
print('Graphique sauvegarde :', OUTPUT_IMG)
plt.show()
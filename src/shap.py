import joblib
import os
from sklearn.preprocessing import LabelEncoder

print("--- Recréation du traducteur de classes (Label Encoder) ---")

# 1. On recrée le dossier 'app' au cas où Colab l'aurait effacé
os.makedirs('app', exist_ok=True)

# 2. On recrée l'encodeur avec les 7 classes exactes de ton dataset
le = LabelEncoder()
classes_obosite = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III',
    'Overweight_Level_I',
    'Overweight_Level_II'
]
le.fit(classes_obosite)

# 3. On sauvegarde le fichier au bon endroit
joblib.dump(le, '/content/app/models/label_encoder.pkl')

print("✅ Fichier 'app/label_encoder.pkl' recréé avec succès !")




import matplotlib.pyplot as plt
import shap
import os
import joblib


explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)  # ← crée la variable shap_values


print("--- Génération du Graphique SHAP Global (Format Barres Lisibles) ---")

# 1. Vérification du dossier 'app'
os.makedirs('app', exist_ok=True)

# 2. Chargement des noms des classes depuis ton encodeur
le = joblib.load('/content/app/models/label_encoder.pkl')
class_names = list(le.classes_)

# 3. Configuration d'une grande figure pour que le texte soit bien aéré
plt.figure(figsize=(12, 8))

# 4. LE SECRET DE LISIBILITÉ : plot_type="bar"
# Si shap_values est une liste (comme avec Random Forest), cela crée un magnifique graphique empilé
# On limite à max_display=10 pour ne garder que le "Top 10" des variables les plus importantes
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    class_names=class_names,
    max_display=10,
    show=False
)

# 5. Ajout d'un titre professionnel
plt.title("Top 10 des variables influençant le risque d'obésité", fontsize=16, pad=20)

# 6. Ajustement des marges pour ne pas couper le texte
plt.tight_layout()

# 7. Sauvegarde en très haute définition (dpi=300)
plt.savefig('/content/app/static/shap_summary_clear.png', bbox_inches='tight', dpi=300)

# Affichage dans Google Colab
plt.show()

print("✅ Graphique lisible sauvegardé sous 'app/shap_summary_clear.png' !")
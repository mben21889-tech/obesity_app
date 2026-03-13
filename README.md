# 🩺 ObesoScan AI - Estimation du Risque d'Obésité

Projet développé dans le cadre de la Coding Week (École Centrale Casablanca). Cet outil clinique d'aide à la décision permet d'estimer le risque d'obésité d'un patient grâce au Machine Learning, avec une transparence totale des décisions via l'explicabilité SHAP.

## 📌 Réponses aux Questions Critiques (Analyse & Modélisation)

### 1. Le dataset était-il équilibré ? Comment avez-vous géré le déséquilibre ?
**Analyse :** Le dataset UCI original présente une distribution relativement bien équilibrée entre les 7 classes d'obésité (chaque classe représente entre ~12% et ~16% des données). 
**Stratégie et Impact :** Bien que le déséquilibre soit mineur, nous avons choisi de le traiter avec une rigueur stricte pour éviter tout biais en faveur des classes majoritaires. Nous avons utilisé l'argument `class_weight='balanced'` lors de l'instanciation de nos modèles (Random Forest, LightGBM). L'impact est que l'algorithme a automatiquement pénalisé les erreurs sur les classes légèrement minoritaires, garantissant un modèle parfaitement équitable sans recourir à un suréchantillonnage artificiel.

### 2. Quel modèle ML a été le plus performant ? Fournissez les métriques.
**Analyse :** Après avoir comparé Random Forest, XGBoost et LightGBM, le modèle basé sur **Random Forest** a été sélectionné comme étant le plus robuste et le plus performant pour ce problème de classification multiclasse.
* **Accuracy :** ~0.94
* **Precision (weighted) :** ~0.94
* **Recall (weighted) :** ~0.94
* **F1-Score (weighted) :** ~0.94
* **ROC-AUC :** ~0.99

### 3. Quelles caractéristiques médicales ont le plus influencé les prédictions (SHAP) ?
**Analyse :** L'intégration de SHAP (SHapley Additive exPlanations) dans notre interface révèle de manière consistante que le **Poids (Weight)** est la variable la plus discriminante. Les autres facteurs majeurs influençant la décision de l'IA sont les **Antécédents familiaux de surpoids** (`family_history_with_overweight`) et la **fréquence d'activité physique** (`FAF`).

### 4. Prompt Engineering : Quelle a été l'utilité des prompts pour la tâche sélectionnée ?
**Tâche ciblée :** Développement de la fonction d'optimisation mémoire `optimize_memory(df)`.
* **Prompt utilisé :** *"Génère une fonction Python robuste nommée optimize_memory(df) pour réduire l'utilisation de la RAM d'un DataFrame Pandas en downcastant les entiers (int64 vers int32/16/8) et les flottants en fonction de leurs valeurs min/max, et en transformant les objets de type string en 'category'. Utilise numpy.iinfo et numpy.finfo. Affiche le pourcentage de mémoire gagnée."*
* **Insights obtenus :** L'utilisation de ce prompt très directif a permis d'obtenir un code prêt pour la production dès la première itération. Nous avons appris qu'en ingénierie de prompt, contraindre l'IA à utiliser des bibliothèques spécifiques (`numpy.iinfo`) l'empêche de proposer des solutions de contournement naïves ou instables, ce qui garantit un code beaucoup plus fiable.

---

## 🚀 Reproductibilité du Projet

Ce projet est entièrement reproductible en suivant ces commandes dans votre terminal :

**1. Installer les dépendances :**
```bash
pip install -r requirements.txt
**2. Lancer l'interface utilisateur (Streamlit) :**
```bash
streamlit run app/app.py
python -m pytest
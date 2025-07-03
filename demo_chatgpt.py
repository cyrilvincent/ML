from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger le jeu de données MNIST
print("Chargement des données MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']

# 2. Normalisation (optionnel mais utile)
X = X / 255.0
y = y.astype(int)

# 3. Division en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entraînement du modèle Random Forest
print("Entraînement du modèle Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 5. Prédiction et évaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy sur le jeu de test : {acc:.4f}")

# 6. Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# 7. Matrice de confusion
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Prédictions")
plt.ylabel("Véritables")
plt.title("Matrice de confusion - Random Forest sur MNIST")
plt.show()

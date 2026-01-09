"""
Régression logistique avec scikit-learn sur le dataset digits
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data
y = digits.target

print(f"\nDimensions des données : {X.shape}")
print(f"Nombre de classes : {len(np.unique(y))}")
print(f"Classes : {np.unique(y)}")

#Affichage du dataset de digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.suptitle("Exemples du dataset digits")
plt.tight_layout()
plt.savefig("digits_examples.png", dpi=150)
plt.show()

# Découpage des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Taille de l'ensemble d'entraînement : {X_train.shape[0]} ({100*X_train.shape[0]/len(X):.1f}%)")
print(f"Taille de l'ensemble de test : {X_test.shape[0]} ({100*X_test.shape[0]/len(X):.1f}%)")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configuration du modèle
model = LogisticRegression(
    max_iter=1000,          # Nombre max d'itérations
    solver='lbfgs',         # Algorithme d'optimisation
    random_state=42
)

# Entraînement
model.fit(X_train_scaled, y_train)
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Précision
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"   - Précision des données d'entraînement : {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"   - Précision des données de test : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(classification_report(y_test, y_pred_test, digits=4))

#Matrice de confusion
cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.colorbar()

classes = digits.target_names
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Afficher les valeurs dans la matrice
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# Visualisation des prédictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

errors_idx = np.where(y_pred_test != y_test)[0]
correct_idx = np.where(y_pred_test == y_test)[0]

# Mélanger les prédictions correctes et incorrectes
n_errors = min(5, len(errors_idx))
n_correct = 10 - n_errors

np.random.seed(42)
selected_correct = np.random.choice(correct_idx, size=n_correct, replace=False)
selected_errors = errors_idx[:n_errors] if n_errors > 0 else []

all_indices = list(selected_correct) + list(selected_errors)
is_error = [False] * n_correct + [True] * n_errors
combined = list(zip(all_indices, is_error))
np.random.shuffle(combined)

for i, ax in enumerate(axes.flat):
    idx, error = combined[i]
    ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    if error:
        ax.set_title(f"Vrai: {y_test[idx]}, Prédit: {y_pred_test[idx]}", color='red')
    else:
        ax.set_title(f"Vrai: {y_test[idx]}, Prédit: {y_pred_test[idx]}", color='green')
    ax.axis('off')

plt.suptitle("Exemples de prédictions (rouge = erreurs, vert = correct)")
plt.tight_layout()
plt.savefig("predictions_examples.png", dpi=150)
plt.show()

print(f"Accuracy finale (test) : {test_accuracy*100:.2f}%")
print(f"Nombre d'erreurs : {len(errors_idx)} / {len(y_test)}")

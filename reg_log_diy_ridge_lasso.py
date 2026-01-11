import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from my_descent import GradientDescent


def sigmoid(z):
    """
    Fonction sigmoïde pour la régression logistique
    """
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, regularization='none', lambda_reg=0.01):
    """
    Calcule le coût de la régression logistique
    """
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-15  # Pour éviter log(0)

    # Coût de base (entropie croisée)
    cost = (-1/m) * (y @ np.log(h + epsilon) + (1 - y) @ np.log(1 - h + epsilon))

    # Ajout du terme de régularisation (on ne régularise pas le biais theta[0])
    if regularization == 'ridge':  # L2 regularization
        reg_term = (lambda_reg / (2 * m)) * np.sum(theta[1:] ** 2)
        cost += reg_term
    elif regularization == 'lasso':  # L1 regularization
        reg_term = (lambda_reg / m) * np.sum(np.abs(theta[1:]))
        cost += reg_term

    return cost


def gradient_function(theta, X, y, regularization='none', lambda_reg=0.01):
    """
    Calcule le gradient de la fonction de coût
    """
    m = len(y)
    h = sigmoid(X @ theta)

    # Gradient de base
    gradient = (1/m) * (X.T @ (h - y))

    # Ajout du terme de régularisation (on ne régularise pas le biais theta[0])
    if regularization == 'ridge':  # L2 regularization
        reg_grad = np.zeros_like(theta)
        reg_grad[1:] = (lambda_reg / m) * theta[1:]
        gradient += reg_grad
    elif regularization == 'lasso':  # L1 regularization
        reg_grad = np.zeros_like(theta)
        reg_grad[1:] = (lambda_reg / m) * np.sign(theta[1:])
        gradient += reg_grad

    return gradient


def fit(X, y, learning_rate=0.01, max_iterations=1000, regularization='none', lambda_reg=0.01):
    """
    A la manière de scikitlearn, on déclare cette fonction fit pour entrainer un modèle de régression logistique
    """

    # La colonne de 1 permet de simuler le biais (intercept)
    m, n = X.shape
    X_bias = np.c_[np.ones(m), X]
    
    # Theta 
    theta_initial = np.zeros(n + 1)
    
    # Le gradient prend theta et retourne le gradient de la fonction de coût
    def gradf(theta):
        """
        Calcule le gradient de la fonction de coût logistique
        """
        return gradient_function(theta, X_bias, y, regularization, lambda_reg)

    # On utilise la descente de gradient de my_descent.py
    gd = GradientDescent(gradient=gradf, learning_rate=learning_rate, max_iterations=max_iterations)
    theta_optimal = gd.descent(initial_point=theta_initial)

    return theta_optimal


def predict_binary(theta, X):
    """
    Prédit les probabilités pour un modèle binaire et retourne les probabilités
    """
    m = X.shape[0]
    X_bias = np.c_[np.ones(m), X]
    return sigmoid(X_bias @ theta)


def fit_multiclass(X, y, learning_rate=0.01, max_iterations=1000, regularization='none', lambda_reg=0.01):
    """
    Entraine un classifieur multiclasse avec la méthode one-vs-rest
    """
    classes = np.unique(y)
    thetas = []

    print(f"Entraînement du modèle One-vs-Rest avec régularisation {regularization} (lambda={lambda_reg}) pour {len(classes)} classes...")

    for i, classe in enumerate(classes):
        print(f"Entraînement du classifieur pour la classe {classe} ({i+1}/{len(classes)})...")

        # 1 si y == classe, 0 sinon
        y_binary = (y == classe).astype(int)
        
        # On entraine le modèle binaire avec la méthode fit définie plus haut
        theta = fit(X, y_binary, learning_rate, max_iterations, regularization, lambda_reg)
        thetas.append(theta)

    print("Entrainement fini")
    return thetas, classes


def predict_multiclass(thetas, classes, X):
    """
    Prédit les classes pour des nouvelles données
    """
    # On calcule les probabilités pour chaque classe
    probas = np.array([predict_binary(theta, X) for theta in thetas]).T
    
    # On choisit la classe avec la probabilité maximale
    predictions = classes[np.argmax(probas, axis=1)]

    return predictions


# Test de la régression logistique avec régularisation
print("Test de la régression logisitque avec régularisation")

# Chargement des données de chiffres
digits = datasets.load_digits()

# Affichage de quelques exemples
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

print("Affichage des exemples d'images...")
plt.show()

# Préparation des données
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Normalisation des données (importante pour la convergence)
data = data / 16.0  # Les pixels sont entre 0 et 16

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=True, random_state=42
)

print(f"\nNombre d'échantillons d'entraînement: {len(X_train)}")
print(f"Nombre d'échantillons de test: {len(X_test)}")
print(f"Nombre de features: {X_train.shape[1]}")

# Test des différents types de régularisation
regularizations = ['none', 'ridge', 'lasso']
lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]  # Ajout de valeurs plus élevées pour voir l'effet

results = {}

for reg in regularizations:
    for lambda_reg in lambdas:
        print(f"Test avec régularisation {reg.upper()} (lambda={lambda_reg})")

        # Entraînement du modèle
        thetas, classes = fit_multiclass(X_train, y_train, learning_rate=0.1, max_iterations=2000,
                                                 regularization=reg, lambda_reg=lambda_reg)

        # Prédictions
        print("\nPrédictions sur l'ensemble de test...")
        predicted = predict_multiclass(thetas, classes, X_test)

        # Évaluation
        accuracy = np.mean(predicted == y_test)
        print(f"\nPrécision sur l'ensemble de test: {accuracy * 100:.2f}%")

        # Rapport de classification détaillé
        print("\nRapport de classification:")
        print(metrics.classification_report(y_test, predicted))

        # Matrice de confusion
        print("\nMatrice de confusion:")
        cm = metrics.confusion_matrix(y_test, predicted)
        print(cm)

        # Affichage en clair de la matrice de confusion
        print("\nMatrice de confusion détaillée:")
        print("Prédit →")
        print("Réel ↓")
        print("      ", end="")
        for classe in classes:
            print(f"{classe:>3}", end="")
        print()
        for i, classe_reel in enumerate(classes):
            print(f"  {classe_reel}  ", end="")
            for j, classe_pred in enumerate(classes):
                print(f"{cm[i, j]:>3}", end="")
            print()

        # Stocker les résultats
        results[f"{reg}_{lambda_reg}"] = {
            'accuracy': accuracy,
            'thetas': thetas,
            'predicted': predicted
        }

print("Récap")

for key, result in results.items():
    reg_type, lambda_val = key.split('_')
    print(f"{reg_type.upper():>6} (lambda={lambda_val:>5}): {result['accuracy']*100:>6.2f}%")

print("Aanalyse de l'évolution des coefficients avec lambda")

# Extraire les coefficients pour chaque méthode et chaque lambda
coeff_evolution = {'ridge': {}, 'lasso': {}}

for key, result in results.items():
    reg_type, lambda_val = key.split('_')
    if reg_type in ['ridge', 'lasso']:
        # Prendre les coefficients du premier classifieur (classe 0)
        # theta[1:] pour exclure le biais
        coeffs = result['thetas'][0][1:]  # Coefficients pour la classe 0
        coeff_evolution[reg_type][float(lambda_val)] = coeffs

# Trier les lambdas pour l'affichage
lambdas_sorted = sorted(coeff_evolution['ridge'].keys())

# Visualisation de l'évolution des coefficients
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Ridge
if coeff_evolution['ridge']:
    ridge_coeffs = np.array([coeff_evolution['ridge'][lam] for lam in lambdas_sorted])
    for i in range(min(10, ridge_coeffs.shape[1])):  # Afficher seulement les 10 premiers coefficients
        ax1.plot(lambdas_sorted, ridge_coeffs[:, i], label=f'Coeff {i}', alpha=0.7)
    ax1.set_xscale('log')
    ax1.set_xlabel('lambda (log scale)')
    ax1.set_ylabel('Valeur des coefficients')
    ax1.set_title('Évolution des coefficients avec Ridge (L2)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

# Lasso
if coeff_evolution['lasso']:
    lasso_coeffs = np.array([coeff_evolution['lasso'][lam] for lam in lambdas_sorted])
    for i in range(min(10, lasso_coeffs.shape[1])):  # Afficher seulement les 10 premiers coefficients
        ax2.plot(lambdas_sorted, lasso_coeffs[:, i], label=f'Coeff {i}', alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_xlabel('lambda (log scale)')
    ax2.set_ylabel('Valeur des coefficients')
    ax2.set_title('Évolution des coefficients avec Lasso (L1)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Comparaison visuelle des précisions
labels = list(results.keys())
accuracies = [results[key]['accuracy'] * 100 for key in labels]

plt.figure(figsize=(14, 6))
bars = plt.bar(labels, accuracies)
plt.xlabel('Type de régularisation et lambda')
plt.ylabel('Précision (%)')
plt.title('Comparaison des précisions avec différents types de régularisation')
plt.xticks(rotation=45)

# Ajouter les valeurs sur les barres
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()
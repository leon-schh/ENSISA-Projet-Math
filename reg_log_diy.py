"""
Original from  https://scikit-learn.org/1.5/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
"""


# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from my_descent import GradientDescent


# Fonction sigmoïde
def sigmoid(z):
    """Fonction sigmoïde pour la régression logistique"""
    return 1 / (1 + np.exp(-z))


# Fonction de coût pour la régression logistique
def cost_function(theta, X, y):
    """
    Calcule le coût de la régression logistique
    
    Paramètres :
    - theta : vecteur des paramètres (poids)
    - X : matrice des features (avec biais)
    - y : vecteur des labels (0 ou 1)
    """
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-15  # Pour éviter log(0)
    cost = (-1/m) * (y @ np.log(h + epsilon) + (1 - y) @ np.log(1 - h + epsilon))
    return cost


# Fonction gradient pour la régression logistique
def gradient_function(theta, X, y):
    """
    Calcule le gradient de la fonction de coût
    
    Paramètres :
    - theta : vecteur des paramètres (poids)
    - X : matrice des features (avec biais)
    - y : vecteur des labels (0 ou 1)
    """
    m = len(y)
    h = sigmoid(X @ theta)
    gradient = (1/m) * (X.T @ (h - y))
    return gradient


# Fonction fit pour entraîner une régression logistique binaire
def fit(X, y, learning_rate=0.01, max_iterations=1000):
    """
    Entraîne un modèle de régression logistique binaire en utilisant la classe GradientDescent
    
    Paramètres :
    - X : matrice des features
    - y : vecteur des labels (0 ou 1)
    - learning_rate : taux d'apprentissage
    - max_iterations : nombre maximal d'itérations
    
    Retourne :
    - theta : vecteur des paramètres optimaux
    """
    # Ajouter une colonne de 1 pour le biais (intercept)
    m, n = X.shape
    X_bias = np.c_[np.ones(m), X]
    
    # Initialiser theta
    theta_initial = np.zeros(n + 1)
    
    # Créer une fonction gradient adaptée pour GradientDescent
    # Le gradient prend theta et retourne le gradient de la fonction de coût
    def gradf(theta):
        """Calcule le gradient de la fonction de coût logistique"""
        return gradient_function(theta, X_bias, y)
    
    # Créer l'objet GradientDescent et effectuer la descente
    gd = GradientDescent(gradient=gradf, learning_rate=learning_rate, max_iterations=max_iterations)
    theta_optimal = gd.descent(initial_point=theta_initial)
    
    return theta_optimal


# Fonction pour prédire avec un modèle binaire
def predict_binary(theta, X):
    """
    Prédit les probabilités pour un modèle binaire
    
    Paramètres :
    - theta : vecteur des paramètres
    - X : matrice des features
    
    Retourne :
    - probabilités pour la classe positive
    """
    m = X.shape[0]
    X_bias = np.c_[np.ones(m), X]
    return sigmoid(X_bias @ theta)


# Fonction pour entraîner un classifieur multiclasse One-vs-Rest
def fit_multiclass(X, y, learning_rate=0.01, max_iterations=1000):
    """
    Entraîne un classifieur multiclasse avec la méthode One-vs-Rest
    
    Paramètres :
    - X : matrice des features
    - y : vecteur des labels (classes multiples)
    - learning_rate : taux d'apprentissage
    - max_iterations : nombre maximal d'itérations
    
    Retourne :
    - Liste des theta pour chaque classe
    - Liste des classes uniques
    """
    classes = np.unique(y)
    thetas = []
    
    print(f"Entraînement du modèle One-vs-Rest pour {len(classes)} classes...")
    
    for i, classe in enumerate(classes):
        print(f"Entraînement du classifieur pour la classe {classe} ({i+1}/{len(classes)})...")
        
        # Créer les labels binaires : 1 si y == classe, 0 sinon
        y_binary = (y == classe).astype(int)
        
        # Entraîner le modèle binaire
        theta = fit(X, y_binary, learning_rate, max_iterations)
        thetas.append(theta)
    
    print("Entraînement terminé!")
    return thetas, classes


# Fonction pour prédire avec le modèle multiclasse
def predict_multiclass(thetas, classes, X):
    """
    Prédit les classes pour des nouvelles données
    
    Paramètres :
    - thetas : liste des paramètres pour chaque classe
    - classes : liste des classes
    - X : matrice des features
    
    Retourne :
    - Vecteur des prédictions
    """
    # Calculer les probabilités pour chaque classe
    probas = np.array([predict_binary(theta, X) for theta in thetas]).T
    
    # Choisir la classe avec la probabilité maximale
    predictions = classes[np.argmax(probas, axis=1)]
    
    return predictions


# Chargement des données
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

# Entraînement du modèle
thetas, classes = fit_multiclass(X_train, y_train, learning_rate=0.1, max_iterations=2000)

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
cm = metrics.confusion_matrix(y_test, predicted)
print("\nMatrice de confusion:")
print(cm)

# Affichage de la matrice de confusion avec matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.colorbar()

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
plt.show()

# Visualisation des poids (thetas) appris pour chaque classifieur OvR
# Chaque theta représente ce que le classifieur "cherche" pour identifier un chiffre
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
for ax, theta, classe in zip(axes.flat, thetas, classes):
    ax.set_axis_off()
    # theta[0] est le biais, theta[1:] sont les poids pour les 64 pixels
    weights = theta[1:].reshape(8, 8)
    # Afficher les poids avec une colormap divergente (bleu=négatif, rouge=positif)
    im = ax.imshow(weights, cmap='RdBu_r', interpolation="nearest")
    ax.set_title(f"Classe {classe}")

plt.suptitle("Poids appris pour chaque classifieur One-vs-Rest\n(Rouge=poids positif, Bleu=poids négatif)")
plt.tight_layout()
plt.colorbar(im, ax=axes, shrink=0.6, label="Poids")
plt.show()

# Affichage de quelques prédictions
_, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
for ax, image, prediction, true_label in zip(axes.flat, X_test[:10], predicted[:10], y_test[:10]):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    color = 'green' if prediction == true_label else 'red'
    ax.set_title(f"Pred: {prediction}\nVrai: {true_label}", color=color)

plt.suptitle("Exemples de prédictions (vert=correct, rouge=erreur)")
plt.tight_layout()
plt.show() 
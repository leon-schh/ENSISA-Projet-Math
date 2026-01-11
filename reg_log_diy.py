import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from my_descent import GradientDescent

def sigmoid(z):
    """
    Fonction sigmoïde pour la régression logistique
    """
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y):
    """
    Calcule le coût de la régression logistique
    """
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-15  # Pour éviter log(0)
    cost = (-1/m) * (y @ np.log(h + epsilon) + (1 - y) @ np.log(1 - h + epsilon))
    return cost

def gradient_function(theta, X, y):
    """
    Calcule le gradient de la fonction de coût
    """

    m = len(y)
    h = sigmoid(X @ theta)
    gradient = (1/m) * (X.T @ (h - y))
    return gradient


def fit(X, y, learning_rate=0.01, max_iterations=1000):
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

        return gradient_function(theta, X_bias, y)
    
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


def fit_multiclass(X, y, learning_rate=0.01, max_iterations=1000):
    """
    Entraine un classifieur multiclasse avec la méthode one-vs-rest
    """

    classes = np.unique(y)
    thetas = []
    
    for i, classe in enumerate(classes):
        print(f"Entraînement du classifieur pour la classe {classe} ({i+1}/{len(classes)})...")
        
        # 1 si y == classe, 0 sinon
        y_binary = (y == classe).astype(int)
        
        # On entraine le modèle binaire avec la méthode fit définie plus haut
        theta = fit(X, y_binary, learning_rate, max_iterations)
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

digits = datasets.load_digits()

# On affiche quelques exemples
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

plt.show()

# Préparation des données comme dans l'exemple donné sur le site de scikit-learn
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# On normalise les données
# Les valeurs de gris sont entre 0 et 16
data = data / 16.0

# Séparation des données en train et test
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

# Rapport
print("\nRapport:")
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

# Visualisation des poids (thetas) appris pour chaque classifieur ovr
# Chaque theta représente ce que le classifieur "cherche" pour identifier un chiffre
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
for ax, theta, classe in zip(axes.flat, thetas, classes):
    ax.set_axis_off()

    # theta[0] est le biais, theta[1:] sont les poids pour les 64 pixels
    weights = theta[1:].reshape(8, 8)
    # On affiche une heatmap des poids
    im = ax.imshow(weights, cmap='RdBu_r', interpolation="nearest")
    ax.set_title(f"Classe {classe}")

plt.suptitle("Poids/Theta appris pour chaque classifieur one-vs-rest\n(Rouge=positif, Bleu=négatif)")
plt.tight_layout()
plt.colorbar(im, ax=axes, shrink=0.6, label="Poids")
plt.show()

# On affiche quelques exemples prédictions
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
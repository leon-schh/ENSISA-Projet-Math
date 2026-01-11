import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.metrics import confusion_matrix

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build a minimal neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(16, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train for 3 epochs
model.fit(X_train, y_train, epochs=3, verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# Visualize sample images from the test dataset
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f"Label: {y_test[i]}")
plt.suptitle("Sample Images from MNIST Test Set")
plt.show()

# Get predictions for a subset of test data
predictions = np.argmax(model.predict(X_test[:5]), axis=1)

# Visualize predictions vs true labels
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f"Pred: {predictions[i]}, True: {y_test[i]}")
plt.suptitle("Predictions vs True Labels")
plt.show()

# Generate predictions for the entire test set
print("\nGénération des prédictions pour l'ensemble de test...")
y_pred = np.argmax(model.predict(X_test), axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de Confusion - Réseau de Neurones sur MNIST')
plt.colorbar()

classes = np.arange(10)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Display values in the matrix
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

# Calculate per-class accuracy
print("\nPrécision par classe:")
for i in range(10):
    class_acc = cm[i, i] / cm[i, :].sum()
    print(f"Classe {i}: {class_acc*100:.2f}%")
# ENSISA - Projet Mathematiques

## Installation

```bash
pip install -r requirements.txt
```

Dependances : numpy, matplotlib, scikit-learn, tensorflow, keras, pandas

## Programmes

### reg_log_diy.py
Implementation from scratch de la regression logistique avec descente de gradient personnalisee.

### reg_log_diy_ridge_lasso.py
Regression logistique avec regularisation Ridge (L2) et Lasso (L1).

### reg_log_sl.py
Comparaison des performances avec scikit-learn.

### doc_reg_log_sl.py
Version provenant de la documentation de scikit-learn sur le datasets digits

### neural_network.py
Reseau de neurones simple sur le dataset MNIST avec Keras

### my_descent.py
Classe GradientDescent pour l'optimisation par descente de gradient, provenant du cours

## Utilisation

Chaque script Python peut etre execute directement :
```bash
python nom_du_script.py
```

Le rapport detaille est disponible dans rapport.pdf
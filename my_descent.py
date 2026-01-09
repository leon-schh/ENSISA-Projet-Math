class GradientDescent:
    def __init__(
        self, gradient, learning_rate: float = 0.01, max_iterations: int = 1000
    ):
        """
        Initialise l'objet GradientDescent avec les paramètres nécessaires.

        Paramètres :
        - gradient : La fonction gradient de la fonction de coût.
        - learning_rate : Taux d'apprentissage (pas) pour la mise à jour des paramètres.
        - max_iterations : Nombre maximal d'itérations de l'algorithme de descente.
        """
        self.gradient = gradient  # tuple de vecteurs (df/dx, df/dy, ...)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def descent(self, initial_point) -> int:
        """
        Effectue l'algorithme de descente de gradient.

        Paramètres :
        - initial_point : Le point de départ de l'algorithme.

        Retourne :
        - Le point optimal trouvé par l'algorithme.
        """
        step = self.learning_rate
        current_point = initial_point
        for i in range(self.max_iterations):
            current_gradient = self.gradient(current_point)
            current_point = self.update(current_point, current_gradient)

            # step = step / (i + 2)  # linearly decreasing
            # step = step / (i + 2) ** 2  # quadratically decreasing
            # step = step * np.exp(-self.beta * (i + 1))  # exponentially decreasing
            # step = step / (self.alpha * (i + 1) + 1)  # keras linearly decreasing

        return current_point

    def update(self, point, gradient_value):
        """
        Met à jour le point en utilisant le gradient et le taux d'apprentissage.

        Paramètres :
        - point : Le point à mettre à jour.
        - gradient_value : Le gradient de la fonction de coût au point donné.

        Retourne :
        - Le nouveau point après la mise à jour.
        """
        return point - self.learning_rate * gradient_value
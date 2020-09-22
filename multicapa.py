import numpy as np

class MulticapaPerceptron(object):
    def __init__(self, no_of_inputs, no_of_hidden=3, no_of_exits=1, threshold=50, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs * no_of_hidden + no_of_hidden * no_of_exits)
        self.bias = 0.01
        self.no_of_hidden = no_of_hidden
        self.no_of_exits = no_of_exits
        
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        if summation >= 0:
          return 1
        else:
          return -1

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # print("Prediction: " + str(prediction) + ", label: " + str(label))
                self.weights += self.learning_rate * (label - prediction) * inputs
                self.bias += self.learning_rate * (label - prediction)
                # print(self.weights);
                # error = calculate_error(inputs, prediction, self.weights)
    def propagation():

    def backPropagation():

    def tanh(x):
        return np.tanh(x)

    # Funcion para obtener la derivada de tanh x
    def dtanh(x):
        return 1.0 - np.tanh(x)**2

    # Funcion sigmoide de x
    def sigmoide(x):
        return 1/(1+np.exp(-x))

    # Funcion para obtener la derivada de de la funcion sigmoide
    def dsigmoide(x):
        s = 1/(1+np.exp(-x))
        return s * (1-s)

    def Datos_entrenamiento(matriz,x1,xn):
        xin = matriz[:,x1:xn+1]
        return xin

    def Datos_validacion(matriz,xji,xjn):
        xjn = matriz[:,xji:xjn+1]
        return xjn

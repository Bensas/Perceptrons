import numpy as np

def tanh(val):
  return np.tanh(val)

def tanh_derivative(val):
  return 1 - (np.tanh(val) ** 2)

class NonLinearPerceptron(object):
    def __init__(self, no_of_inputs, threshold=50, learning_rate=0.01, activation_function=tanh, activation_function_derivative=tanh_derivative):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs)
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights)
        return self.activation_function(summation)

    def predict_derivative(self, inputs):
        summation = np.dot(inputs, self.weights)
        return self.activation_function_derivative(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                derivative = self.predict_derivative(inputs)
                # print("Prediction: " + str(prediction) + ", label: " + str(label))
                self.weights += self.learning_rate * (label - prediction) * derivative * inputs
                # self.bias += self.learning_rate * (label - prediction)
                # print(self.weights);
                # error = calculate_error(inputs, prediction, self.weights)
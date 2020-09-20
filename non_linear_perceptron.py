import numpy as np

def tanh(val):
  return np.tanh(val)

def tanh_derivative(val):
  return 1 - (np.tanh(val) ** 2)

def sigmoid(activation, beta=0.5):
	return 1 / (1 + np.exp(-2*beta*activation))

def sigmoid_derivate(activation, beta=0.5):
	return 2 * beta * sigmoid(activation, beta=beta) * (1 - sigmoid(activation, beta=beta))

class NonLinearPerceptron(object):
    def __init__(self, no_of_inputs, threshold=100, learning_rate=1, activation_function=sigmoid, activation_function_derivative=sigmoid_derivate, scale=100):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs)
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.scale = scale
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights)
        return self.activation_function(summation)

    def predict_derivative(self, inputs):
        summation = np.dot(inputs, self.weights)
        return self.activation_function_derivative(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.scale * self.predict(inputs)
                derivative = self.scale * self.predict_derivative(inputs)
                # print(self.weights)
                # print(prediction)
                # print(derivative)
                # print(label)
                # print("Prediction: " + str(prediction) + ", label: " + str(label))
                self.weights += self.learning_rate * (label - prediction) * derivative * inputs
                print(self.cost_function(training_inputs, labels))
                # self.bias += self.learning_rate * (label - prediction)
                # error = calculate_error(inputs, prediction, self.weights)

    def cost_function(self, inputs, labels):
      error = 0.0
      for inputs, label in zip(inputs, labels):
        prediction = self.scale * self.predict(inputs)
        error += (prediction - label) ** 2
      return error
import numpy as np

def excercise1():
  x = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
  y = [-1, -1, -1, 1]

  x = np.array(x)
  y = np.array(y)

  print(x)
  print(y)
  perceptron = Perceptron(2)
  perceptron.train(x, y)
  print(perceptron.weights)
  # print(np.dot(perceptron.weights, [-1, 1]))

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[0:] += self.learning_rate * (label - prediction) * inputs


excercise1()
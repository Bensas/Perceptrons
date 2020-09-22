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

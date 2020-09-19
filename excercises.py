import numpy as np
from simple_perceptron import SimplePerceptron

def excercise1():
  x = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
  y = [-1, -1, -1, 1]

  x = np.array(x)
  y = np.array(y)

  print("=== OR FUNCTION ===")
  print("X=" + str(x))
  print("Y=" + str(y))
  print("Training...")
  perceptron = SimplePerceptron(2)
  perceptron.train(x, y)
  print("Resulting weights: " + str(perceptron.weights))
  print_perceptron_test(perceptron, x, y)

  print("=== XOR FUNCTION ===")
  y = [1, 1, -1, -1]
  y = np.array(y)
  print("X=" + str(x))
  print("Y=" + str(y))
  print("Training...")
  perceptron = SimplePerceptron(2)
  perceptron.train(x, y)
  print("Resulting weights: " + str(perceptron.weights))
  print_perceptron_test(perceptron, x, y)


def excercise2():
  x = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
  y = [-1, -1, -1, 1]

  x = np.array(x)
  y = np.array(y)

  print("=== OR FUNCTION ===")
  print("X=" + str(x))
  print("Y=" + str(y))
  print("Training...")
  perceptron = SimplePerceptron(2)
  perceptron.train(x, y)
  print("Resulting weights: " + str(perceptron.weights))
  print_perceptron_test(perceptron, x, y)

  # print("=== XOR FUNCTION ===")
  # y = [1, 1, -1, -1]
  # y = np.array(y)
  # print("X=" + str(x))
  # print("Y=" + str(y))
  # print("Training...")
  # perceptron = Perceptron(2)
  # perceptron.train(x, y)
  # print("Resulting weights: " + str(perceptron.weights))
  # print(perceptron.predict([-1, 1]) > 1)
  # print(perceptron.predict([1, -1]) > 1)
  # print(perceptron.predict([-1, -1]) < 0)
  # print(perceptron.predict([1, 1]) < 0)

def print_perceptron_test(perceptron, inputs, expected_outputs):
  print("Perceptron tests (all should be true):")
  for curr_input, curr_expected_output in zip(inputs, expected_outputs):
    print(perceptron.predict(curr_input) == curr_expected_output)


excercise1()
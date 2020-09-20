import csv
import numpy as np
from simple_perceptron import SimplePerceptron
from non_linear_perceptron import NonLinearPerceptron

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
  x = []
  f = open('ej2_X.txt', 'r')
  reader = csv.reader(f, delimiter=' ', lineterminator='\n')
  for row in reader:
      x_row = []
      for each in row:
          if(each != ''):
              x_row.append(float(each))
      x.append(x_row)
  f.close()

  y = []
  f = open('ej2_Y.txt', 'r')
  reader = csv.reader(f, delimiter=' ', lineterminator='\n')
  for row in reader:
      for each in row:
          if (each != ''):
              y.append(float(each))
  f.close()

  x = np.array(x)
  y = np.array(y)

  # print("=== OR FUNCTION ===")
  print("X=" + str(x))
  print("Y=" + str(y))
  print("Training...")
  perceptron = NonLinearPerceptron(3)
  perceptron.train(x, y)
  print(perceptron.cost_function(x, y))
  # print_perceptron_test(perceptron, x, y)
  print("Resulting weights: " + str(perceptron.weights))

def excercise3():
  x = [[-1, 1], [1, -1], [-1, -1], [1, 1]]
  y = [1, 1, -1, -1]

  x = np.array(x)
  y = np.array(y)

def print_perceptron_test(perceptron, inputs, expected_outputs):
  print("Perceptron tests (all should be true):")
  for curr_input, curr_expected_output in zip(inputs, expected_outputs):
    print(perceptron.predict(curr_input) == curr_expected_output)


excercise2()

# 4.47, -4.08, 4.45 -> 87

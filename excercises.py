import csv
import numpy as np
from numpy import random
from simple_perceptron import SimplePerceptron
from non_linear_perceptron import NonLinearPerceptron
from multicapa import MLP

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
  # x = normalize_arr(x)
  y = normalize_arr(y)
  print("Normalized X=" + str(x))
  print("Normalized Y=" + str(y))
  print("Training...")
  perceptron = NonLinearPerceptron(3)
  perceptron.train(x, y)
  # print(perceptron.cost_function(x, y))
  # print_perceptron_test(perceptron, x, y)
  print("Resulting weights: " + str(perceptron.weights))

def Datos_entrenamiento(matriz,x1,xn):
    xin = matriz[:,x1:xn+1]
    return xin

def excercise3():
  x = [[-1, 1, 1], [1, -1, 1], [-1, -1, -1], [1, 1, -1]]

  matrix_data = np.array(x)
  # Datos de entrada
  x_inicio = 0
  x_n = 1
  # Crear vector de entradas xi
  xi = (Datos_entrenamiento(matrix_data,x_inicio,x_n))
  d = matrix_data[:,x_n+1]
  # Parametros de la red
  f,c = xi.shape
  fac_ap = 0.2
  precision = 0.00000001
  epocas = 10000 #
  epochs = 0
  # Arquitectura de la red
  n_entradas = c # numero de entradas
  cap_ocultas = 1 # Una capa oculta
  n_ocultas = 6 # Neuronas en la capa oculta
  n_salida = 1 # Neuronas en la capa de salida
  # Valor de umbral o bia
  us = 1.0 # umbral en neurona de salida
  uoc = np.ones((n_ocultas,1),float) # umbral en las neuronas ocultas
  # Matriz de pesos sinapticos
  random.seed(0) # 
  w_1 = random.rand(n_ocultas,n_entradas)
  w_2 = random.rand(n_salida,n_ocultas)
    
  #Inicializar la red PMC
  print(w_1)
  print(w_2)
  print("despues ... \n")
  red = MLP(xi,d,w_1,w_2,us,uoc,precision,epocas,fac_ap,n_ocultas,n_entradas,n_salida)
  epochs,w1_a,w2_a,us_a,uoc_a,E = red.Aprendizaje()
  print(w1_a)
  print(w2_a)
  print("error final: ")
  print(red.error_red)

def print_perceptron_test(perceptron, inputs, expected_outputs):
  print("Perceptron tests (all should be true):")
  for curr_input, curr_expected_output in zip(inputs, expected_outputs):
    print(perceptron.predict(curr_input) == curr_expected_output)

def normalize_arr(arr):
  min = np.min(arr)
  max = np.max(arr)    
  return (arr - min) / (max - min) 

excercise3()

# 4.47, -4.08, 4.45 -> 87

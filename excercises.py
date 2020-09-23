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

  print("=== AND FUNCTION ===")
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

  find_optimal_training_group(x, y)

  # train_x = np.array(x[:99])
  # train_y = np.array(y[:99])
  # validate_x = np.array(x[100:])
  # validate_y = np.array(y[100:])

  # print("Training set X=" + str(train_x))
  # print("Training set Y=" + str(train_y))
  # train_y = normalize_arr(train_y)
  # validate_y = normalize_arr(validate_y)
  # print("Normalized Y=" + str(train_y))
  # print("Training...")
  # perceptron = NonLinearPerceptron(no_of_inputs=3, threshold=200)
  # perceptron.train(train_x, train_y)
  # print("Resulting weights: " + str(perceptron.weights))
  # print("Validation set cost function: " + str(perceptron.cost_function(validate_x, validate_y)))
  # print_perceptron_test(perceptron, x, y)

def find_optimal_training_group(x, y):
  current_min_error = 1000
  min_i, min_j = 0, 0
  for i in range(len(y)):
    for j in range(i, len(y)):
      train_x = np.array(x[i:j+1])
      train_y = np.array(y[i:j+1])

      validate_x = x[:i]
      validate_x.extend(x[j+1:])
      validate_x = np.array(validate_x)

      validate_y = y[:i]
      validate_y.extend(y[j+1:])
      validate_y = np.array(validate_y)

      train_y = normalize_arr(train_y)
      validate_y = normalize_arr(validate_y)
      perceptron = NonLinearPerceptron(no_of_inputs=3, threshold=200)
      perceptron.train(train_x, train_y)
      error = perceptron.cost_function(validate_x, validate_y)
      # print(error + " " + str(i) + " - " + str(j))
      print(error)
      print(i)
      print(j)
      if error < current_min_error:
        current_min_error = error
        min_i, min_j = i, j
  print("Minimum error was " + str(current_min_error) + " for training set indexes " + str(min_i) + ", " + str(min_j))


def Datos_entrenamiento(matriz,x1,xn):
    xin = matriz[:,x1:xn+1]
    return xin

def excercise3():

  # Ejercicio 3.1

  x = [[-1, 1, 1], [1, -1, 1], [-1, -1, -1], [1, 1, -1]]

  # Ejercicio 3.2

  # x = []
  # i = 0
  # x_row = []
  # # utilizamos una version acortada de los pixels para obtener los pesos ideales (con un error muy chico).
  # f = open('ej3_mapa_pixels_acortado.txt', 'r')
  # reader = csv.reader(f, delimiter=' ', lineterminator='\n')
  # for row in reader:
  #     for each in row:
  #         if (each != ''):
  #             i = i + 1
  #             x_row.append(int(each))
  #         if(i == 36):
  #             i = 0
  #             x.append(x_row)
  #             x_row = []
  # f.close()

  matrix_data = np.array(x)
  # Datos de entrada
  x_inicio = 0
  x_n = 34
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

  # Esta parte es para el ejercicio 3.2, aca prueba testear si el algoritmo te devuelve si el numero es par o impar del 0 al 9
  
  # print("\nprobando los nuevos pixeles: \n")

  # x = []
  # i = 0
  # x_row = []
  # f = open('ej3_mapa_pixels.txt', 'r')
  # reader = csv.reader(f, delimiter=' ', lineterminator='\n')
  # for row in reader:
  #     for each in row:
  #         if (each != ''):
  #             i = i + 1
  #             x_row.append(int(each))
  #         if(i == 36):
  #             i = 0
  #             x.append(x_row)
  #             x_row = []
  # f.close()
  # matrix_data = np.array(x)
  # d = matrix_data[:,x_n+1]
  # xi = (Datos_entrenamiento(matrix_data,x_inicio,x_n))
  # epocas = 0
  # red2 = MLP(xi,d,w1_a,w2_a,us_a,uoc_a,precision,epocas,fac_ap,n_ocultas,n_entradas,n_salida)
  # red2.Aprendizaje()

def print_perceptron_test(perceptron, inputs, expected_outputs):
  print("Perceptron tests (all should be true):")
  for curr_input, curr_expected_output in zip(inputs, expected_outputs):
    print(perceptron.predict(curr_input) == curr_expected_output)

def normalize_arr(arr):
  if len(arr) == 0:
    return arr
  min = np.min(arr)
  max = np.max(arr)    
  return (arr - min) / (max - min) 

excercise2()

# 4.47, -4.08, 4.45 -> 87

# Funcion para obtener la tanh
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
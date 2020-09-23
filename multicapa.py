import numpy as np
from numpy import random
import matplotlib.pyplot as plt

class MLP():
    # constructor
    def __init__(self,all_inputs,labels,w_1,w_2,bias,uoc,precision,epocas,learning_rate,n_ocultas,n_entradas,n_salida):
        # Variables de inicialización 
        self.all_inputs = np.transpose(all_inputs)
        self.labels = labels
        self.w1 = w_1
        self.w2 = w_2
        self.bias = bias
        self.uoc = uoc
        self.precision = precision
        self.epocas = epocas
        self.learning_rate = learning_rate
        self.n_entradas = n_entradas
        self.n_ocultas = n_ocultas
        self.n_salida = n_salida
        # Variables de aprendizaje
        self.current_label = 0 # Salida deseada en iteracion actual
        self.error_red = 1 # Error total de la red en una conjunto de iteraciones
        self.Ew = 0 # Error cuadratico medio
        self.prev_error = 0 # Error anterior
        self.Errores = []
        self.Error_actual = np.zeros(len(labels)) # Errores acumulados en un ciclo de muestras
        self.current_inputs = np.zeros((1,n_entradas))
        self.hidden_layer_inputs = np.zeros((n_ocultas,1)) # Entradas en neuronas ocultas
        self.hidden_layer_outputs = np.zeros((n_ocultas,1)) # Resultado de la activacion en neuronas ocultas
        self.output_layer_input = 0.0 # Entrada en la neurona de salida
        self.y = 0.0 # Resultado de la activación en la neurona de salida
        self.current_epochs = 0
        # Variables de retropropagacion
        self.error_real = 0
        self.output_delta = 0.0 # delta de salida
        self.hidden_output_delta = np.zeros((n_ocultas,1)) # Deltas en neuronas ocultas
        
    def Operacion(self):
        respuesta = np.zeros((len(self.labels),1))
        for p in range(len(self.labels)):
            self.current_inputs = self.all_inputs[:,p]
            self.Propagar()
            respuesta[p,:] = self.y
        return respuesta.tolist()
    
    def Aprendizaje(self):
        Errores = [] # Almacenar los errores de la red en un ciclo
        while(np.abs(self.error_red) > self.precision):
            self.prev_error = self.Ew
            for i in range(len(self.labels)):
                self.current_inputs = self.all_inputs[:,i] # Senales de entrada por iteracion
                self.current_label = self.labels[i]
                self.Propagar()
                self.Backpropagation()
                self.Propagar()
                print("esperado: ")
                print(self.current_label)
                print("resultado: ")
                print(self.y)
                print("\n")
                self.Error_actual[i] = (0.5)*((self.current_label - self.y)**2)
            # error global de la red
            self.Error()
            Errores.append(self.error_red)
            self.current_epochs +=1
            # Si se alcanza un mayor numero de epocas
            if self.current_epochs > self.epocas:
                break
        # Regresar 
        return self.current_epochs,self.w1,self.w2,self.bias,self.uoc,Errores
    
    # def Test(self):    
    #     self.current_inputs = self.all_inputs[:,34] # Senales de entrada por iteracion
    #     self.Propagar()
    #     self.Backpropagation()
    #     self.Propagar()
    #     print("resultado: ")
    #     print(self.y)
                
    
    def Propagar(self):
        # Operaciones en la primer capa
        for a in range(self.n_ocultas):
            self.hidden_layer_inputs[a,:] = np.dot(self.w1[a,:], self.current_inputs) + self.uoc[a,:]
        
        # Calcular la activacion de la neuronas en la capa oculta
        for o in range(self.n_ocultas):
            self.hidden_layer_outputs[o,:] = tanh(self.hidden_layer_inputs[o,:])
        
        # Calcular Y potencial de activacion de la neuronas de salida
        self.output_layer_input = (np.dot(self.w2,self.hidden_layer_outputs) + self.bias)
        # Calcular la salida de la neurona de salida
        self.y = tanh(self.output_layer_input)
    
    def Backpropagation(self):

        self.error_real = (self.current_label - self.y)

        self.output_delta = (dtanh(self.output_layer_input) * self.error_real)

        self.w2 = self.w2 + (np.transpose(self.hidden_layer_outputs) * self.learning_rate * self.output_delta)

        self.bias = self.bias + (self.learning_rate * self.output_delta)

        self.hidden_output_delta = dtanh(self.hidden_layer_inputs) * np.transpose(self.w2) * self.output_delta

        for j in range(self.n_ocultas):
            self.w1[j,:] = self.w1[j,:] + ((self.hidden_output_delta[j,:]) * self.current_inputs * self.learning_rate)
        
        # Ajustar el umbral en las neuronas ocultas
        for g in range(self.n_ocultas):
            self.uoc[g,:] = self.uoc[g,:] + (self.learning_rate * self.hidden_output_delta[g,:])
        
    def Error(self):
        # Error cuadratico medio
        self.Ew = ((1/len(self.labels)) * (sum(self.Error_actual)))
        self.error_red = (self.Ew - self.prev_error)

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
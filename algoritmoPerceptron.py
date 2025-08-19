########## Algoritmo de Perceptron##########
import numpy as np
import random

# ------> DEFINIR VARIABLES <------
#Componentes del vector de entrada
i=0
# Vectores de entrada 
i_x = 0
#Dimensiones del plano en R^n
N= 2
# Numero de datos de entrada
p = 11
#Cota de iteraciones -> Numero máximo de iteraciones
COTA = 1000
#Tasa de aprendizaje >>>Como se calcula?
n=0.1
# Salida deseada >>>Como se calcula?
y = 0
#Valores de los errores
error = 1
error_min = p*2


# Matriz de datos de entrada -> Conjunto de entrenamiento
x = np.random.rand((p, N))
# Pesos de entrada -> Crear la matiz de pesos -> Pesos sinapticos
w = np.zeros(N+1,1)

#Funcion para calcular el error
def CalcularError(x, y, w, p):
    return error


while error > 0 and i < COTA: 
    # Definir i_x al azar entre 1 y p
    i_x = random.rand(1, p)
    # Calcular la exitacion 
    h = x[i_x].w
    # Calcular la activacion  O = signo(h) 
    if h > 0: 
        O = 1
    else:
        O = -1
    #Calcular el ajuste de pesos
    w_ajuste = n*(y[i_x] - O.x[i_x])
    #Ajustar los pesos
    w = w + w_ajuste
    #Cual es la funcion? >>>Como se calcula?
    error = CalcularError(x, y, w,p)
    if error < error_min:
        error_min = error
        error_min_w = w
    i += 1

# Funcion logica "Y"

import numpy as np

def CalcularError(y, oTotal):
  error = 0.5 * np.sum((y - oTotal) ** 2)
  print("Error = " , error)
  return error

def signo(h):
  if h >= 0:
    return 1
  else:
    return -1

tasaAprendizaje = 0.1
p = 4
w_min = 0
epoca = 0
w = np.array([1, 1, 1])
error = 1
error_min = 10

entrada = np.array([[-1,1,1], [1,-1,1], [-1,-1,1], [1,1,1]])
y = np.array([[-1], [-1], [-1], [1]])
w = np.array([1, 1, 1])

while error > 0 and epoca < (100):

  oTotal = np.zeros((4,1))
  print(oTotal)

  for i in range (p):
    print("Iteracion: ", i)
    h = entrada[i] @ w
    print("Entradas: ", entrada[i])
    print(h)
    O = signo(h)
    print("Salida del perceptron: ", O)
    oTotal[i][0] = O
    print("Salida arreglo: ", oTotal)
    deltaW =tasaAprendizaje * (y[i] - O) * entrada[i]
    w = w + deltaW
    print("Pesos: ", w)

  error = CalcularError(y, oTotal)
  print(error)

  if error < error_min:
    print("Nuevo error minimo")
    error_min = error
    w_min = w.copy()
    print("W_min = " , w_min)

  epoca = epoca+ 1
  print("epoca " ,epoca)
  print("W_min = " , w_min)
  print("Error_ min = " , error_min)
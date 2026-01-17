import numpy as np

# -------> DEFINIR VARIABLES <------
# Variable de iteracion 
i = 0
#Numero de dimensiones del plano en R^n
N = 2
#Numero de datos de entrada
p=4
#Defrinir COTA -> Numero de iteraciones 
COTA = 100
#Definir el arreglo para los datos de entrada -> Con el umbral
x = np.array([[-1, 1, 1], [1, -1, 1], [-1, -1, 1], [1, 1, 1]])
# Definir la salida deseada
y = np.array([-1, -1, -1, 1])
# Definir la tasa de aprendizaje
n = 0.1
h=0

# ---> Formulaciones del modelo <------

# Definir los pesos iniciales 
w = np.zeros([N+1])
print("Pesos iniciales: ", w)

#Definir el error 
error = 1
error_min = (p*2)+2

#Funcion para la activacion 
def signo(h):
    if h >= 0:
        return 1
    else:
        return -1
    
def CalcularError( y, O):
    error = 0.5 * np.sum((y- O)**2)
    print("Error: ", error)
    return error


print("Datos de entrada: ")
print(x)
print("Salida deseada: ", y)
    
# ---> Modelo de Perceptron <----
i = 0

while error > 0 and i < COTA:

    O_list = []
    for j in range(p):
        print("Iteracion: ", j)
        idx = np.random.randint(0, p)  # Generar índice aleatorio entre 0 y p-1
        i_x = x[idx]
        print("Entradas: ", i_x)
        i_y = y[idx]
        print("Índice aleatorio: ", idx)
        print("Arreglo", i)
        h = i_x @ w
        print("Funcion exitacion")
        print(h)

        #Calcular la activacion O = signo(h)
        O = signo(h)
        O_list.append(O)
        print("Salida del perceptron: ", O)
        print("Salida esperada: ", y[j])

        # Calcular el ajuste de pesos para que O = y
        w_ajuste = (n * (y[j] - O) * x[j])
        print ("Ajuste de pesos resta: ", (y[j] - O))
        print("Ajuste de pesos: multiplicar",(y[j] - O) * x[j] )
        
        print("Ajuste de pesos: resultado " , (n * (y[j] - O) * x[j]))
        print("Ajuste de pesos: ")
        print(w_ajuste)

        #Ajustar los pesos
        w = w + w_ajuste
        print("Pesos actualizados: ")
        print(w)
        print("--------------------------------------------------")

    # Calcular el error después de procesar todos los patrones
    O_array = np.array(O_list)
    error = CalcularError(np.array(y), O_array)
    print("Error: ", error)

    if error < error_min:
        print("Nuevo error minimo encontrado: ", error)
        error_min = error
        print("Error minimo actualizado: ", error_min)
        error_min_w = w
        print("Pesos para el error minimo: ", error_min_w)
    i += 1
print("Proceso terminado")

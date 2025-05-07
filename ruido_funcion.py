import numpy as np

# Función para agregar ruido gaussiano a una imagen
def agregar_ruido(imagen, sigma):
    ruido = np.random.normal(0, sigma, imagen.shape)
    imagen_ruidosa = np.clip(imagen + ruido, 0, 255)
    return imagen_ruidosa

# Función objetivo J(u) y su gradiente
def funcion_objetivo(u, f, lambda_reg):
    return 0.5 * np.sum((u - f) ** 2) + 0.5 * lambda_reg * np.sum(np.gradient(u)[0]**2 + np.gradient(u)[1]**2)

def gradiente_funcion_objetivo(u, f, lambda_reg):
    grad_f = u - f
    grad_laplaciano = lambda_reg * (np.gradient(np.gradient(u)[0])[0] + np.gradient(np.gradient(u)[1])[1])
    return grad_f + grad_laplaciano

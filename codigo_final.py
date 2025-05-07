import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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

# Métodos de descenso de gradiente

def descenso_gradiente_momentum(f, grad_f, x0, alpha, beta, lambda_reg, max_iter, eps):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    for i in range(max_iter):
        grad = grad_f(x, f, lambda_reg)  # Pasamos f (imagen ruidosa) y lambda_reg a grad_f
        norma_grad = np.linalg.norm(grad)
        if norma_grad < eps:
            break  
        v = beta * v + (1 - beta) * grad
        x = x - alpha * v
    return x

def descenso_gradiente_simple(f, grad_f, x0, alpha, lambda_reg, max_iter, eps):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = grad_f(x, f, lambda_reg)  # Pasamos f (imagen ruidosa) y lambda_reg a grad_f
        norma_grad = np.linalg.norm(grad)
        if norma_grad < eps:
            break  
        x = x - alpha * grad
    return x

def descenso_gradiente_nesterov(f, grad_f, x0, alpha, beta, lambda_reg, max_iter, eps):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)  # Inicializar el momento a cero
    for i in range(max_iter):
        y = x + beta * v  # Predicción de la siguiente posición
        grad = grad_f(y, f, lambda_reg)  # Calculamos el gradiente en la posición predicha
        norma_grad = np.linalg.norm(grad)
        if norma_grad < eps:
            break
        v = beta * v + grad  # Actualización del momento
        x = x - alpha * v  # Actualización de la posición

    return x


# Cargar imagen original y agregarle ruido
image = cv2.imread('/Users/josuearteaga/Documents/Ibero/4to semestre/Optimización matemática/1.jpg', cv2.IMREAD_GRAYSCALE)  # Asegúrate de tener la imagen en el mismo directorio
image = cv2.resize(image, (256, 256))  # Redimensionar la imagen si es necesario

# Convertir la imagen a un rango de 0 a 1
image = image.astype(float) / 255.0

# Agregar diferentes niveles de ruido
sigma_values = [0.1, 0.05, 0.010]  # Correción ruido
lambda_reg = 0.1  # Parámetro de regularización

# Métodos de optimización
alpha = 0.00001 #bajamos alpha como dios manda
beta = 0.006 #beta tmb
max_iter = 500
eps = 1e-5

# Comparar los resultados
for sigma in sigma_values:
    noisy_image = agregar_ruido(image, sigma)
    
    # Funciones de descenso de gradiente
    print(f"Descenso de gradiente simple (Ruido sigma={sigma})")
    denoised_simple = descenso_gradiente_simple(noisy_image, gradiente_funcion_objetivo, noisy_image, alpha, lambda_reg, max_iter, eps)
    
    print(f"Descenso de gradiente con momentum (Ruido sigma={sigma})")
    denoised_momentum = descenso_gradiente_momentum(noisy_image, gradiente_funcion_objetivo, noisy_image, alpha, beta, lambda_reg, max_iter, eps)
    
    print(f"Descenso de gradiente de Nesterov (Ruido sigma={sigma})")
    denoised_nesterov = descenso_gradiente_nesterov(noisy_image, gradiente_funcion_objetivo, noisy_image, alpha, beta, lambda_reg, max_iter, eps)
    
    # Calcular PSNR y SSIM
    ssim_simple = ssim(image, denoised_simple, data_range=1.0)
    psnr_simple = psnr(image, denoised_simple, data_range=1.0)

    ssim_momentum = ssim(image, denoised_momentum, data_range=1.0)
    psnr_momentum = psnr(image, denoised_momentum, data_range=1.0)

    ssim_nesterov = ssim(image, denoised_nesterov, data_range=1.0)
    psnr_nesterov = psnr(image, denoised_nesterov, data_range=1.0)
        
    # Mostrar resultados visuales
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(noisy_image, cmap='gray')
    plt.title(f'Imagen Ruidosa (Sigma={sigma})')
    
    plt.subplot(2, 2, 2)
    plt.imshow(denoised_simple, cmap='gray')
    plt.title(f'Denoising Simple PSNR={psnr_simple:.2f}, SSIM={ssim_simple:.2f}')
    
    plt.subplot(2, 2, 3)
    plt.imshow(denoised_momentum, cmap='gray')
    plt.title(f'Denoising Momentum PSNR={psnr_momentum:.2f}, SSIM={ssim_momentum:.2f}')
    
    plt.subplot(2, 2, 4)
    plt.imshow(denoised_nesterov, cmap='gray')
    plt.title(f'Denoising Nesterov PSNR={psnr_nesterov:.2f}, SSIM={ssim_nesterov:.2f}')
    
    plt.tight_layout()
    plt.show()
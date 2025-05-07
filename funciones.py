import numpy as np

def descenso_gradiente_momentum(f, grad_f, x0, alpha, beta, max_iter, eps):
    x = np.array(x0, dtype=float)  
    v = np.zeros_like(x)  
    for i in range(max_iter):
        grad = grad_f(x)  
        norma_grad = np.linalg.norm(grad)  
        if norma_grad < eps:
            break  
        v = beta * v + (1 - beta) * grad  
        x = x - alpha * v  
        print(f"{i}\t,{x}\t,{norma_grad}")
    return x

def descenso_gradiente_simple(f,grad_f,x0,alpha,max_iter,eps):
    x_historico = [x0]
    for i in range(max_iter):
        f_i = f(x0)
        grad_f_i = grad_f(x0)
        norma_grad = np.linalg.norm(grad_f_i)
        if norma_grad < eps:
            break
        xi = x0 - alpha*grad_f_i
        x0 = xi.copy()
        print(f"{i}\t,{x0}\t,{norma_grad}")
    return x0


def descenso_gradiente_nesterov(f, grad_f, x0, alpha, beta, max_iter, eps):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    
    for i in range(max_iter):
        y = x + beta * v  # Predicción de la posición
        grad = grad_f(y)  # Gradiente en la posición predicha
        norma_grad = np.linalg.norm(grad)  # Norma del gradiente
        
        if norma_grad < eps:
            break
        
        v = beta * v + grad  # Actualización del momento
        x = x - alpha * v  # Actualización de la posición
        
        print(f"{i}\t,{x}\t,{norma_grad}")
    
    return x

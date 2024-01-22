import numpy as np
import matplotlib.pyplot as plt

# Funkcja tanh
def tanh(x):
    return np.tanh(x)

# Gradient funkcji tanh
def tanh_gradient(x):
    tanh_x = tanh(x)
    return 1 - tanh_x**2

# Zakres danych x
x = np.linspace(-7, 7, 200)

# Obliczamy wartości funkcji tanh i jej gradientu
tanh_values = tanh(x)
tanh_gradient_values = tanh_gradient(x)

# Tworzymy wykres
plt.figure(figsize=(8, 6))
plt.plot(x, tanh_values, label='tanh')
plt.plot(x, tanh_gradient_values, label='Gradient tanh')
plt.legend()
plt.xlabel('x')
plt.ylabel('Wartosc')
plt.title('Funkcja tanh i jej Gradient')
plt.grid(True)
plt.show()

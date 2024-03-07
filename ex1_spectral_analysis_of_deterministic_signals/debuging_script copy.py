import numpy as np
from numpy.fft import ifft

mu = np.array([6, 4, 4, 5, 3, 4, 5, 0, 0, 0, 0])
N = len(mu)  # Długość tablicy mu

k = np.arange(N)
A = 10

tmpmu = 2-1/2  # DFT eigenfrequency worst case
tmpmu = 1  # DFT eigenfrequency best case

x = A * np.exp(tmpmu * +1j*2*np.pi/N * k)

X_ = np.zeros((N, 1), dtype=complex)
for mu_ in range(N):
    for k_ in range(N):
        X_[mu_] += x[k_] * np.exp(-1j*2*np.pi/N*k_*mu_)

x_ = np.zeros((N, 1), dtype=complex)
for k_ in range(N):
    for mu_ in range(N):
        x_[k_] += X_[mu_] * np.exp(+1j*2*np.pi/N*k_*mu_)
x_ *= 1/N

mu_values = np.arange(N)
K = np.outer(k, mu_values)
W = np.exp(+1j * 2*np.pi/N * K)

# Zaktualizuj X_test i x_test zgodnie z tablicą mu
X_test = np.array([A * np.exp(tmpmu * +1j*2*np.pi/N * mu_val) for mu_val in mu])
x_test = 1/N * np.matmul(W, X_test)

print(np.allclose(ifft(X_test), x_test))
print('DC is 1 as expected: ', np.mean(x_test))

x_test2 = np.sum([X_test[i] * W[:, i] for i in range(N)], axis=0)
x_test2 *= 1/N
print(np.allclose(x_test, x_test2))

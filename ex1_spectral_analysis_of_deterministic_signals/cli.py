import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft

def main():
    input_array = []
    print("Defining Xmu table ('f' for finish)")
    while True:
        print(input_array)
        ipt = input(">> ")
        if ipt == "f": break
        try:
            input_array.append(int(ipt))
        except:
            print("It is not a number.")

    mu = np.array(input_array)
    N = len(mu)  # Długość tablicy mu
    k = np.arange(N)
    A = 10

    print("getting all possible entries k")
    K = np.outer(k, np.arange(N))
    print(K)

    print("Fourier matrix setting up")
    W = np.exp(+1j * 2*np.pi/N * K)
    print(W)

    fig, ax = plt.subplots(1, N)
    fig.set_size_inches(6, 6)
    fig.suptitle(
        r'Fourier Matrix for $N=$%d, blue: $\Re(\mathrm{e}^{+\mathrm{j} \frac{2\pi}{N} \mu k})$, orange: $\Im(\mathrm{e}^{+\mathrm{j} \frac{2\pi}{N} \mu k})$' % N)

    for tmp in range(N):
        ax[tmp].set_facecolor('lavender')
        ax[tmp].plot(W[:, tmp].real, k, 'C0o-', ms=7, lw=0.5)
        ax[tmp].plot(W[:, tmp].imag, k, 'C1o-.', ms=7, lw=0.5)
        ax[tmp].set_ylim(N-1, 0)
        ax[tmp].set_xlim(-5/4, +5/4)
        if tmp == 0:
            ax[tmp].set_yticks(np.arange(0, N))
            ax[tmp].set_xticks(np.arange(-1, 1+1, 1))
            ax[tmp].set_ylabel(r'$\longleftarrow k$')
        else:
            ax[tmp].set_yticks([], minor=False)
            ax[tmp].set_xticks([], minor=False)
        ax[tmp].set_title(r'$\mu=$%d' % tmp)
    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    fig.savefig('fourier_matrix.png', dpi=300)
    print("Figure: fourrier_matrix is generated.")

    X_test = mu
    print("Processing matrix multiplication...")
    x_test = 1/N * np.matmul(W, X_test)
    print("testing... (test 1)")
    if not np.allclose(ifft(X_test), x_test):
        raise ValueError("Something went wrong... (test 1)")
    print("test 1 ok")


    print("applying linear combination of the Fourrier matrix...")
    x_test2 = np.sum([X_test[i] * W[:, i] for i in range(N)], axis=0)
    x_test2 *= 1/N
    print("testing... (test 2)")

    if not np.allclose(x_test, x_test2):
        raise ValueError("Something went wrong... (test 2)")
    print("test 2 ok")

    plt.figure(figsize=(10,5), dpi=300)
    plt.stem(k, np.real(x_test), label='real',
            markerfmt='C0o', basefmt='C0:', linefmt='C0:')
    plt.stem(k, np.imag(x_test), label='imag',
            markerfmt='C1o', basefmt='C1:', linefmt='C1:')  
    
    # note that connecting the samples by lines is actually wrong, we
    # use it anyway for more visual convenience

    plt.plot(k, np.real(x_test), 'C0o-', lw=0.5)
    plt.plot(k, np.imag(x_test), 'C1o-', lw=0.5)
    plt.xlabel(r'sample $k$')
    plt.ylabel(r'$x[k]$')
    plt.legend()
    plt.grid(True)
    plt.savefig("IDFT_result.png")
    print("Figure: IDFT_result is generated.")

if __name__ == "__main__":
    main()

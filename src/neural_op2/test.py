
import numpy as np
from scipy.sparse import diags, identity, kron

def test_laplacian(n):
    n_int = n - 2

    e = np.ones(n_int)
    T = diags([e[:-1], -2 * e, e[:-1]], offsets=[-1, 0, 1])
    I = identity(n_int)

    L2D = kron(I, T) + kron(T, I)
    return -L2D.tocsr().toarray()

def discrete_laplacian(n):
    if (n <= 2):
        raise Exception("Matrix not needed. No interior points.")

    n_int = (n-2)**2
    m = int(np.sqrt(n_int)) 
    A = np.zeros((n_int,n_int))
    
    # diagonal
    for i in range(n_int):
        for j in range(n_int):
            if (i==j):
                A[i,j] = 4
    
    # the upper diagonal
    for k in range(m):
        for i in range(k*m, (k+1)*m-1):
            A[i,i+1] = -1
    
    # the lower diagonal
    for k in range(m):
        for i in range(k*m+1,(k+1)*m):
            A[i,i-1] = -1
    
    # identity matrices
    for i in range(m,n_int):
        A[i-m,i] = -1
        A[i,i-m] = -1

    return A

def main():
    n = 3 # n = (N - 2)
    N = n + 2
    x = 1
    e = np.ones(n)
    T = diags([e[:-1], -2 * e, e[:-1]], offsets=[-1, 0, 1]).toarray()
    I = identity(n)

    assert n > 1, "when n = 1 then A is just a scalar [[4]]"
    g = np.ones(4)
    A_1 = diags([g[:-2],g[:-1], -4 * g, g[:1],g[:2]], offsets=[-2, -1, 0, 1, 2]).toarray()
    
    A = kron(I, T) + kron(T, I)
    print(f"Matrix I:\n{I.toarray()}",
          f"\nMatrix A of shape {A.shape} when N = {N}:\n{A.toarray()[0,:]}",
          f"\nMatrix A_1 of shape {A_1.shape} when N = {N}:\n{A_1}"
          )

if __name__ == "__main__":
    main()

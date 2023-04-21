import numpy as np
from numpy import linalg as LA
import sys
from funcs import *

np.set_printoptions(linewidth=200, threshold=sys.maxsize)

a = int(input('a = '))
b = int(input('b = '))
c = int(input('c = '))
d = int(input('d = '))

n = int(input('n = '))
m = int(input('m = '))

h1 = (b-a)/n
k1 = (d-c)/m

h = (n/(b-a))**2
k = (m/(d-c))**2
mdiag_elem = -2*(h+k)

x = [a + i * h for i in range(n + 1)]
y = [c + i * k for i in range(m + 1)]

print(f'x = {x}, y = {y}')
print(f'1/h**2 = {h}, 1/k**2 = {k}, mdiag_elem = {mdiag_elem}, h1 = {h1}, k1 = {k1}')

def fill_bounds_v(v, x, y):
    for i in range(n + 1):
        v[i, 0] = u_test(x[i], 1)
        v[i, -1] = u_test(x[i], -1)
    for j in range(m + 1):
        v[0, j] = u_test(1, y[j])
        v[-1, j] = u_test(-1, y[j])
        
def fill_rhsf(v):
    F = np.empty((n-1)*(m-1)) # type: ignore
    k = 0
    for j in range(1, m):
        for i in range(1, n):
            if j == 1:
                if i == 1:
                    F[k] = -rsf_test(i, j) - h * v[i - 1, j] - k * v[i, j - 1]
                    k += 1
                elif i == n - 1:
                    F[k] = -rsf_test(i, j) - h * v[i + 1, j] - k * v[i, j - 1]
                    k += 1
                else:
                    F[k] = -rsf_test(i, j) - k * v[i, j - 1]
                    k += 1

            elif j == m - 1:
                if i == 1:
                    F[k] = -rsf_test(i, j) - h * v[i - 1, j] - k * v[i, j + 1]
                    k += 1
                elif i == n - 1:
                    F[k] = -rsf_test(i, j) - h * v[i + 1, j] - k * v[i, j + 1]
                    k += 1
                else:
                    F[k] = -rsf_test(i, j) - k * v[i, j + 1]
                    k += 1

            else:
                if i == 1:
                    F[k] = -rsf_test(i, j) - h * v[i - 1, j]
                    k += 1
                elif i == n - 1:
                    F[k] = -rsf_test(i, j) - h * v[i + 1, j]
                    k += 1
                else:
                    F[k] = -rsf_test(i, j)
                    k += 1
    return F

def fill_matrix():
    matrix = np.diag(np.full((n-1)*(m-1),mdiag_elem)) + np.diag(np.full((n-1)*(m-1) - 1,h), 1) + \
     np.diag(np.full((n-1)*(m-1) - 1,h), -1) + np.diag(np.full((n-1)*(m-1) - 4,k), 4) + \
         np.diag(np.full((n-1)*(m-1) - 4,k), -4)
    
    # matrix = np.zeros(((n-1)*(m-1),(n-1)*(m-1)))
    # np.fill_diagonal(matrix, mdiag_elem)
          
    # for i in range(0,(n-1)*(m-1) - 1):
    #     matrix[i][i+1] = h
        
    # for i in range(1,(n-1)*(m-1)):
    #     matrix[i][i-1] = h
        
    # for i in range(0,(n-1)*(m-1) - 4):
    #     matrix[i][i+4] = k
        
    # for i in range(4,(n-1)*(m-1)):
    #     matrix[i][i-4] = k
        
    return matrix

print(fill_matrix())

def upper_relaxation(A, b, w=1.2):
    flag = False
    Nmax = 1000
    S = 0
    eps = 1e-7
    eps_max = 0
    eps_cur = 0
    
    while not flag:
        eps_max = 0
        for i in range(n):
            x_old = x[i]
            x_new = (1-w)*A[i][i]*x[i]+w*b[i]
            for j in range(m):
                if j != i:
                    x_new -= w*A[i][j]*x[j]
                    x_new /= A[i][i]
                    eps_cur = abs(x_old - x_new)
                    if eps_cur > eps_max:
                        eps_max = eps_cur
                    x[i] = x_new
                S+=1
                if eps_max < eps or S >= Nmax:
                    flag = True
                    
def optimal_w(A):
    # A = L + D + R   
    D = LA.inv(np.diagflat(np.diag(A)))
    ro = max(LA.eigvals(D @ (np.tril(A) - np.triu(A))))
    w = 2/(1+math.sqrt((1-ro**2)))
    return w
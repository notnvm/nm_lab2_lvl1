import numpy as np
from funcs import *

np.set_printoptions(linewidth=100)

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
    F = np.empty((n-1)*(m-1))
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
    matrix = np.zeros(((n-1)*(m-1),(n-1)*(m-1)))
    np.fill_diagonal(matrix, mdiag_elem)
    
    for i, j in zip(range(0, matrix[0].size - (n - 1)), range(n - 1, np.size(matrix, 1))):
        matrix[i][j] = matrix[j][i] = k
        
    rows = [i for i in range((n - 1) * (m - 1))]
    columns = [i + 1 for i in range(0, (n - 1) * (m - 1) - 1)]
    for row, col in zip(rows, columns):
        if row % (n - 1) == n - 2:
            matrix[row][col] = 0
            matrix[col][row] = 0
        else:
            matrix[row][col] = h
            matrix[col][row] = h
    matrix = np.around(matrix, 2)
        
    return matrix

print(fill_matrix())
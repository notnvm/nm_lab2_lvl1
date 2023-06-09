import math
import sys
import numpy as np
from numpy import linalg as LA

class data_values():
    main_task: bool = False
    solved: bool = False
    a: int = 1 
    b: int = 2
    c: int = 2
    d: int = 3
    n: int = 5
    m: int = 5
    eps: float = 1e-6
    nmax: int = 500

np.set_printoptions(linewidth=200, threshold=sys.maxsize)

dv = data_values()

class Funcs():
    def u_test(self, x, y):
        return math.sin(math.pi*x*y)

    def rsf_test(self, x, y):
        return math.sin(math.pi*x*y)*((math.pi**2)*x**2+(math.pi**2)*y**2)

    def mu1_test(self, y):
        return self.u_test(dv.a, y)

    def mu2_test(self, y):
        return self.u_test(dv.b, y)

    def mu3_test(self, x):
        return self.u_test(x, y=dv.c)

    def mu4_test(self, x):
        return self.u_test(x, y=dv.d)
    
    def rsf(self, x, y):
        return -math.exp(-x*y**2)
    
    def mu1(self, y):
        return (y-2)*(y-3)
    
    def mu2(self, y):
        return y*(y-2)*(y-3)
    
    def mu3(self, x):
        return (x-1)*(x-2)
    
    def mu4(self, x):
        return x*(x-1)*(x-2)

func = Funcs()

def find_exact_solution(x, y, n, m):
    u = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            u[i][j] = func.u_test(x[i],y[j])
    return u

def find_matrix_elements(n, m):
    h = (n/(dv.b-dv.a))**2
    k = (m/(dv.d-dv.c))**2
    mdiag_elem = -2*(h+k)
    return h,k,mdiag_elem

def fill_bounds_v(v, x, y, n, m):
    if not dv.main_task:
        for j in range(m + 1):
            v[0, j] = func.mu1_test(y[j])
            v[-1, j] = func.mu2_test(y[j])
        for i in range(n + 1):
            v[i, 0] = func.mu3_test(x[i])
            v[i, -1] = func.mu4_test(x[i])
    else:   
        for j in range(m + 1):
            v[0, j] = func.mu1(y[j])
            v[-1, j] = func.mu2(y[j])
        for i in range(n + 1):
            v[i, 0] = func.mu3(x[i])
            v[i, -1] = func.mu4(x[i])

def fill_matrix(n, m): 
    h,k,mdiag_elem = find_matrix_elements(n, m)
    matrix = np.diag(np.full((n-1)*(m-1),mdiag_elem)) + np.diag(np.full((n-1)*(m-1) - 1,h), 1) + \
     np.diag(np.full((n-1)*(m-1) - 1,h), -1) + np.diag(np.full((n-1)*(m-1) - 4,k), 4) + \
         np.diag(np.full((n-1)*(m-1) - 4,k), -4)
         
    return matrix

def upper_relaxation(v, n, m, w=1.236068):
    flag = False
    Nmax = dv.nmax
    S = 0
    eps = dv.eps
    eps_max = 0
    eps_cur = 0
    x = []
    y = []
    if not dv.main_task:
        x = np.linspace(dv.a, dv.b, dv.n+1)
        y = np.linspace(dv.c, dv.d, dv.m+1)
    else:
        x = np.linspace(dv.a, dv.b, dv.n*2+1)
        y = np.linspace(dv.c, dv.d, dv.m*2+1)
    
    h2 = -(n / (dv.b - dv.a)) ** 2
    k2 = -(m / (dv.d - dv.c)) ** 2
    a2 = -2 * (h2 + k2)
    
    while not flag:
        eps_max = 0
        for j in range(1, m):
            for i in range(1, n):
                v_old = v[i][j]
                v_new = -w*(h2 * (v[i + 1][j] + v[i - 1][j]) + k2 * (v[i][j + 1] + v[i][j - 1]))
                if not dv.main_task:
                    v_new += (1-w)*a2*v[i][j]+w*func.rsf_test(x[i],y[j])
                else:
                    v_new += (1-w)*a2*v[i][j]+w*func.rsf(x[i],y[j])
                v_new /= a2
                eps_cur = math.fabs(v_old - v_new)
                if eps_cur > eps_max:
                    eps_max = eps_cur
                v[i][j] = v_new
        S = S + 1
        if eps_max < eps or S >= Nmax:
            flag = True
    return v, S, eps_max, w

def optimal_w(A):
    # A = L + D + R   
    D = LA.inv(np.diagflat(np.diag(A)))
    ro = max(LA.eigvals(D @ (np.tril(A) - np.triu(A))))
    w = 2/(1+math.sqrt((1-ro**2)))
    return w
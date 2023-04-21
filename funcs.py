import math

def u_test(x, y):
    return math.sin(math.pi*x*y)

def rsf_test(x, y):
    return (math.pi**2)*math.sin(math.pi*x*y)*(x**2+y**2)

def mu1_test(y):
    return u_test(1, y)

def mu2_test(y):
    return u_test(2, y)

def mu3_test(x):
    return u_test(x, 2)

def mu4_test(x):
    return u_test(x, 3)

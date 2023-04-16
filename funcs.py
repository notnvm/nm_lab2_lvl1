import math

def u_test(x, y):
    return math.sin(math.pi*x*y)

def rsf_test(x, y):
    return (math.pi**2)*math.sin(math.pi*x*y)*(x**2+y**2)

def mu1_test(a, y):
    return u_test(a, y)

def mu2_test(b, y):
    return u_test(b, y)

def mu3_test(x, c):
    return u_test(x, c)

def mu4_test(x, d):
    return u_test(x, d)

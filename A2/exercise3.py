import numpy as np
import math

beta = 0.25
alpha = 2

def PDF(alpha, beta, x):
    pdf = beta * alpha * math.exp(-beta * (x ** alpha)) * (x ** (alpha - 1))
    return pdf

def CDF(x):
    cdf = 1 - math.exp(-0.25 * (x ** 2))
    return cdf

print(1 - CDF(4))
print(CDF(4))
print(CDF(1))
print(CDF(10) - CDF(5))


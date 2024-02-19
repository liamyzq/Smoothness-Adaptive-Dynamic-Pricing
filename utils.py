from math import log, floor

def psi(p,beta):
    if abs(p) <= 1:
        return (1 - abs(p)) ** beta
    else:
        return 0

def from_gur(p,beta,M,m,L,C):
    if p >=0 and p <= 0.5:
        value = 0.25*(1+L*(0.5)**beta-L*(p**beta))
    else:
        value = 0.25
        for j in range(1,m+1):
            value += ((-1)**j) * ((2*M)**(-beta)) * C * psi(2*M*(2-2*p)-0.5*(j+0.5),beta)
    return value

    
def f(p,type="from_gur",T=20000):
    if type == "from_gur":
        T = T
        beta = 0.8 # Set the smoothness parameter to 0.8
        alpha = 0.01
        tau1 = 0.8
        L1 = 1
        C1 = 1
        M = (1 / 16) * floor(0.25 * (2 * log(2) / T) ** (-tau1 / (2 * tau1 + 1))) ** (1 / beta)
        m = floor(M ** (1 - alpha * beta))
        value = from_gur(p, beta, M, m, L1, C1)
    else:
        value = None
    if value >= 0:
        return min(value, 1)
    else:
        return 0
        
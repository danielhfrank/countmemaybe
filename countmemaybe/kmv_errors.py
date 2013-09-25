from scipy import optimize, special
import math

def find_k(desired_eps, confidence=0.98, D=0):
    objective = lambda k: relative_error(int(k), confidence, D) - desired_eps
    return int(optimize.brentq(objective, 1e02, 1e08))

def relative_error(k, confidence=0.98, D=0):
    if D:
        u = lambda D, k, e : (k - 1.0) / ((1.0 - e) * D)
        l = lambda D, k, e : (k - 1.0) / ((1.0 + e) * D)
        objective = lambda e, D, k, confidence : special.betainc(k, D-k+1, u(D, k, e)) - special.betainc(k, D-k+1, l(D, k, e)) - confidence

        p = optimize.brentq(objective, 1e-05, 0.5, args=(D, k, confidence), maxiter=300)
    else:
        p = math.sqrt(2.0 / (math.pi * (k - 2)))
    return p

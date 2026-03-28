import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iminuit import Minuit
from iminuit import cost
from scipy import stats
from scipy.integrate import quad
r = np.random

def GEV(x, mu, sigma, xi):
    x = np.asarray(x)
    t = 1 + xi * (x - mu) / sigma
    pdf = np.zeros_like(x, dtype=float)
    valid = t > 0
    pdf[valid] = ((1 / sigma)* t[valid]**(-1/xi - 1)* np.exp(-t[valid]**(-1/xi)))
    return pdf

def negLLHfunc(mu, sigma, xi):
    pdfVals = GEV(times, mu, sigma, xi)
    return -np.sum(np.log(pdfVals))
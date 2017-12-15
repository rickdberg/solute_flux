# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:30:22 2017

@author: rickdberg
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def conc_curve(z,a,b,c):
    return b * np.exp(np.multiply(a,z)) + c # a = (v - sqrt(v**2 * 4Dk))/2D

def cost(params):
    lamb, mu, sigm = params
    model = conc_curve(concunique[:,0],lamb,mu,sigm)
    reg = abs(lamb/np.mean((model - concunique[:,1])**2))**2  # very simple: higher parameters -> higher cost
    regweight = 0.01  # determines relative importance of regularization vs goodness of fit
    return np.mean((model - concunique[:,1])**2)


def concentration_fit(concunique, dp):
    a0 = -0.1
    b0 = concunique[0,1]/5
    c0 = concunique[0,1]
    if np.mean(concunique[1:dp,1]) <= concunique[0,1]:
        res = optimize.minimize(cost, [a0,b0,c0])
        conc_fit = res.x
    else:
        res = optimize.minimize(cost, [-a0,-b0,c0])
        conc_fit = res.x
    # conc_interp_depths = np.arange(0,3,intervalthickness)  # Three equally-spaced points
    # conc_interp_fit = conc_curve(conc_interp_depths, conc_fit)  # To be used if Boudreau method for conc gradient is used
    return conc_fit


conc_fit = concentration_fit(concunique, dp)

conc_interp_fit_plot = conc_curve(np.linspace(concunique[0,0], concunique[dp-1,0], num=50), *conc_fit)




plt.plot(concunique[:,1], concunique[:,0], 'go')
plt.plot(concunique[0:dp,1],
         concunique[0:dp,0],
         'bo', label="Used for curve fit")
plt.plot(conc_interp_fit_plot,
         np.linspace(concunique[0,0],
                     concunique[dp-1,0],num=50),
                     'k-', label="Curve fit", linewidth=2)
plt.show()


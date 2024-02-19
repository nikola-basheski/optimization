#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 23:40:25 2024

@author: nikola
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:16:40 2024

@author: nikola
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 21:41:08 2024

@author: nikola
"""

# Import dependencies
import time
import math
import numpy as np
import pandas as pd
import datetime
import scipy as sc
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from IPython.display import display, Latex
from statsmodels.graphics.tsaplots import plot_acf


#%%

ARI = pd.read_csv(r"/home/nikola/Pictures/ARI.csv")



FBRT = pd.read_csv(r"/home/nikola/Pictures/BXMT (1).csv")

#%%

#ARI = ARI.iloc[-557:, :]


#%%

pearson_corr = ARI['Open'].corr(FBRT['Open'], method='spearman')

#%%
Ari_sample = pd.Series(list(ARI['Close']))#.iloc[-len(ARI):-780]))

Fbrt_sample = pd.Series(list(FBRT['Close']))#.iloc[-len(ARI):-780]))


#%%

pearson_corr = Ari_sample.corr(Fbrt_sample, method='pearson')

print(pearson_corr)


#%%
data1 = Ari_sample.copy()
data2 = Fbrt_sample.copy()

#%%


df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Reset indices
df1_reset = df1.reset_index(drop=True)
df2_reset = df2.reset_index(drop=True)

# Specify the correlation threshold
correlation_threshold = 0

# Set the window size for the rolling correlation
window_size = 40  # Adjust as needed

# Create a DataFrame to store the results
result_df = pd.DataFrame(columns=['Start Index', 'End Index', 'Correlation'])

# Iterate through the rolling window
for i in range(len(df1_reset) - window_size + 1):
    df1_window = Ari_sample.iloc[i:i + window_size]#.values
    df2_window = Fbrt_sample.iloc[i:i + window_size]#.values
    current_correlation = df1_window.corr(df2_window, method='pearson')
    #print(df1_window)
    print(current_correlation)
    # Check if the correlation falls below the threshold
    if current_correlation < correlation_threshold:
        result_df = result_df.append({
            'Start Index': i,
            'End Index': i + window_size - 1,
            'Correlation': current_correlation
        }, ignore_index=True)

# Display the result DataFrame
print(result_df)


#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


#%%
x1  = Ari_sample
x2 = Fbrt_sample

# Create a DataFrame
df = pd.DataFrame({'x1': x1, 'x2': x2})

# Calculate rolling correlation with a window size of 30
rolling_corr = df['x1'].rolling(window=40).corr(df['x2'])

# Find the index where the correlation is below 0.5
breakpoints = rolling_corr[rolling_corr < 0].index

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['x1'], label='Variable 1', linewidth=2)
plt.plot(df['x2'], label='Variable 2', linewidth=2)

# Highlight regions with low correlation
for i in range(0, len(breakpoints)-1, 2):
    plt.axvspan(breakpoints[i], breakpoints[i + 1], color='yellow', alpha=0.3)

# Add labels and legend
plt.xlabel('Months from 2023')
plt.ylabel('Values')
plt.title('Overlay of Two Variables with Low Correlation Highlighted')
plt.legend()
plt.show()
#%%
#k0 = np.mean(df['x1']/df['x2'])
spread = df['x1']-df['x2']

#%%

log_returns = np.log(spread/spread.shift(1)).dropna()

#%%
log_returns.plot()
plt.title('daily log returns')
plot_acf(log_returns)
plt.show()

#%%

log_returns_sq = np.square(log_returns)
log_returns_sq.plot()
plt.title('square daily log returns')
plot_acf(log_returns_sq)
plt.show()

#%%

TRADING_DAYS = 40
volatility = log_returns.rolling(window=TRADING_DAYS).std()*np.sqrt(252)
volatility = volatility.dropna()
volatility.plot()
plt.title('Stock Spread')
plt.show()


#%%
    
vol = np.array(volatility)

#%%


#%%

def mu_func(d, dt, kappa, theta):
    ekt = np.exp(-kappa*dt)
    return d*ekt + theta*(1-ekt)

def std_func(dt, kappa, sigma):
    e2kt = np.exp(-2*kappa*dt)
    return sigma*np.sqrt((1-e2kt)/(2*kappa))
 
    
#%%
kappa=1
theta=3
sigma=1 

vol_dt = vol[1:]
vol_t = vol[:-1]

dt = 1/len(Ari_sample)

mu_OU = mu_func(vol_t, dt, kappa, theta)
sigma_OU = std_func(dt, kappa, sigma)

#l_theta_hat = np.sum( np.log( sc.stats.norm.pdf(x_dt, loc=mu_OU, scale=sigma_OU) ) )
from sympy import log

#%%
from sympy import symbols, Symbol, log, diff, exp, pi, sqrt

def get_log(expr):
    x_, mu, sigma = symbols('x mu sigma')
    return log(expr)
#(1/(2*np.pi))*np.exp(-((x-mu)**2)/(2*(sigma**2)))

def orig_func(x_, mu, sigma):
    expr =  (1/(2*np.pi))*np.exp(-((x_-mu)**2)/(2*(sigma**2)))
    return get_log(expr)

x_ = 1
mu = 3
sigma = 1

orig_func(x_, mu, sigma)

#%%
def obj_func(vol, mu=mu_OU, sigma=sigma_OU):
    res = []
    for i in range(len(vol)):
        res.append(orig_func(vol[i], mu[i], sigma))
        
    return res
func_res = obj_func(vol_dt, mu_OU, sigma_OU)
#%%

def dfunc(vol_t):
    dLambda = np.zeros(len(vol_t))
    h = 1e-3 # this is the step size used in the finite difference.
    for i in range(len(vol_t)):
        dX = np.zeros(len(vol_t))
        dX[i] = h
        dLambda[i] = (orig_func(vol_t[i]+h,mu_OU[i], sigma_OU)-orig_func(vol_t[i]-h,mu_OU[i], sigma_OU))/(2*h);
    return dLambda


#%%

dfunc(vol_t)

#%%


def kappa_pos(theta_hat):
    kappa = theta_hat[0]
    return kappa

def sigma_pos(theta_hat):
    sigma = theta_hat[2]
    return sigma



#%%
from sympy import symbols, Symbol, log, diff, exp, pi, sqrt, simplify

 #%%

def obj_func(theta_hat):
    x, y, z = theta_hat[0], theta_hat[1], theta_hat[2]
    kappa, theta, sigma, dt, x_dt_, x_t_ = symbols('kappa theta sigma dt x_dt_ x_t_')

    x_dt = vol[1:]
    x_t = vol[:-1]

    dt = 1/len(Ari_sample)
    
    
    mu_OU = x_t_*exp(-kappa*dt)+ theta*(1-exp(-kappa*dt))
    sigma_OU = sigma*sqrt((1-exp(-2*kappa*dt))/(2*kappa))
    
    expr =  (1/(2*pi))*exp(-((x_dt_-mu_OU)**2)/(2*(sigma_OU**2)))
    expr_simple = simplify(expr)
    
    dexpr = log(expr_simple).evalf().subs({kappa:x,
                                        theta:y,
                                        sigma:z})

    l_theta_hat = sum(dexpr.subs({x_dt_: x_dt[i],
                                        x_t_:x_t[i]}) for i in range(len(x_dt)))
    
    return -l_theta_hat


#%%
x, y, z = symbols('x y z')
kappa, theta, sigma, dt, x_dt_, x_t_ = symbols('kappa theta sigma dt x_dt_ x_t_')



#%%


def func(theta_space):
    x = theta_space[0]
    #y = theta_space[1]
    z = theta_space[2]
    L1 = theta_space[3]
    L2 = theta_space[4]
    return obj_func(theta_space) + L1*(x-theta_space[0]) + L2*(z-theta_space[2])

#%%
def dfunc(X):
    dLambda = np.zeros(len(X))
    h = 1e-3 # this is the step size used in the finite difference.
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (func(X+dX)-func(X-dX))/(2*h);
    return dLambda

#%%
#from scipy.optimize import fsolve

#X1 = fsolve(dfunc, x0=[1, 3, 1, 0, 0])
    

#%%
    

#%%
def obj_func(theta_hat, vol=vol):
    x, y, z = theta_hat[0], theta_hat[1], theta_hat[2]
    kappa, theta, sigma, dt, x_dt_, x_t_ = symbols('kappa theta sigma dt x_dt_ x_t_')
    mu_OU, sigma_OU = symbols('mu_OU sigma_OU')
    x_dt = vol[1:]
    
    x_t = vol[:-1]

    dt = 1 / 252
    
    mu_OU = x_t_ * exp(-kappa * dt) + theta * (1 - exp(-kappa * dt))
    sigma_OU = sigma * sqrt((1 - exp(-2 * kappa * dt)) / (2 * kappa))
    
    expr = (1 / (2 * pi)) * exp(-((x_dt_ - mu_OU) ** 2) / (2 * (sigma_OU ** 2)))
    epsilon = 1e-10 
    l_theta_hat = sum([log(expr).subs({kappa: x, theta: y, sigma: z, x_dt_: x_dt[i], x_t_: x_t[i]}) for i in range(len(x_t))])
    
    return -l_theta_hat


# Vectorized computation of l_theta_hat
l_theta_hat = obj_func([x, y, z])

# Display result
print(l_theta_hat)


#%%
from sympy.solvers import solve


    #%%
kappa, theta, sigma= symbols('kappa theta sigma')
const_value = 24
x, y, z, lambda0, lambda1 = symbols('x y z lambda0 lambda1', real=True)
gradF = [simplify(diff(obj_func([x,y,z])), kappa), simplify(diff(obj_func([x, y, z]), theta)), simplify(diff(obj_func([x, y, z]), sigma))]


#%%

def loglik_grad(parameters, gradF):
    grad_x = gradF[0].subs({x: parameters[0], y: parameters[1], z: parameters[2]}).evalf()
    grad_y = gradF[1].subs({x: parameters[0], y: parameters[1], z: parameters[2]}).evalf()
    grad_z = gradF[2].subs({x: parameters[0], y: parameters[1], z: parameters[2]}).evalf()
    
    grad = [grad_x, grad_y, grad_z]
    return obj_func(parameters), np.array(grad)

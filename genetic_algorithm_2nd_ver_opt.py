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




import numpy as np

def obj_func(theta_hat, vol = vol):
    x, y, z = theta_hat
    mu_OU = vol[:-1] * np.exp(-x * (1 / 252)) + y * (1 - np.exp(-x * (1 / 252)))
    sigma_OU = z * np.sqrt((1 - np.exp(-2 * x * (1 / 252))) / (2 * x))

    x_dt = vol[1:]
    expr = (1 / (2 * np.pi)) * np.exp(-((x_dt - mu_OU) ** 2) / (2 * (sigma_OU ** 2)))
    epsilon = 1e-10 
    l_theta_hat = -np.sum(np.log(expr+epsilon))

    return l_theta_hat

#%%

def gradient_obj_func(theta_hat, vol = vol):
    x, y, z = theta_hat
    mu_OU = vol[:-1] * np.exp(-x * (1 / 252)) + y * (1 - np.exp(-x * (1 / 252)))
    sigma_OU = z * np.sqrt((1 - np.exp(-2 * x * (1 / 252))) / (2 * x))

    x_dt = vol[1:]
    expr = (1 / (2 * np.pi)) * np.exp(-((x_dt - mu_OU) ** 2) / (2 * (sigma_OU ** 2)))

    grad_x = np.sum(((x_dt - mu_OU) / (sigma_OU ** 2)) * expr)
    grad_y = np.sum((1 - np.exp(-1 / 252)) * (expr / sigma_OU))
    grad_z = np.sum((((x_dt - mu_OU) ** 2) / (sigma_OU ** 3) - 1) * expr)

    return -np.array([grad_x, grad_y, grad_z])

# Initialize parameters and other variables
parameters = np.array([1.0, 3.0, 1.0])


import random
    
#%%

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
NUM_GENERATIONS = 100

# OneMax Problem Parameters
STRING_LENGTH = 20


#%%

individual_gene = random.randint(0,10)


#%%
individual = [random.randint(-10, 10) for i in range(10)]


def gen_individual(lower, higher, num_genes):
    individual = [random.randint(lower, higher) for i in range(num_genes)]
    return individual

def gen_population(POP_SIZE, lower, higher, num_genes):
    population = []
    for k in range(POP_SIZE):
        population.append(gen_individual(lower, higher, num_genes))
    return population

def get_parents(population):
    parent1 = random.choice(population)
    parent2 = random.choice(population)
    while parent1 == parent2:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            
    return parent1, parent2


#%%
import numpy as np

def get_genes_from_parent(parent):
    indices = np.arange(0, len(parent)-1)
    indices = list(indices)
    indices_used = []
    for j in range(int(len(parent)/2)):
        index_chosen = random.choice(indices)
        indices_used.append(index_chosen)
        indices.remove(index_chosen)
    return [parent[i] for i in indices_used]

def get_genes_from_parent_more(parent):
    indices = np.arange(0, len(parent)-1)
    indices = list(indices)
    indices_used = []
    for j in range(math.ceil(len(parent)/2)):
        index_chosen = random.choice(indices)
        indices_used.append(index_chosen)
        indices.remove(index_chosen)
    return [parent[i] for i in indices_used]
        
#%%
def crossover_random(parent1, parent2, num_children):
    
    children = []    
    
    for i in range(num_children):
        if(len(parent1) % 2==0):
            child = get_genes_from_parent(parent1) + get_genes_from_parent(parent2)
            while child in children:
                child = get_genes_from_parent(parent1) + get_genes_from_parent(parent2)    
            children.append(child)
        else:
            parents = [parent1, parent2]
            parent_indices = [0, 1]
            
            parent_index = random.choice(parent_indices)
            first_genes = get_genes_from_parent(parents[parent_index])
            remaining_index = [x for x in parent_indices if x not in [parent_index]][0]
            #print(remaining_index)
            
            second_genes = get_genes_from_parent_more(parents[remaining_index])
            
            child = first_genes + second_genes
            while child in children:
                parents = [parent1, parent2]
                parent_indices = [0, 1]
            
                parent_index = random.choice(parent_indices)
                first_genes = get_genes_from_parent(parents[parent_index])
                remaining_index = [x for x in parent_indices if x not in [parent_index]][0]
                #print(remaining_index)
                second_genes = get_genes_from_parent_more(parents[remaining_index])
            
                child = first_genes + second_genes
                
            children.append(child)
                
            
    return children


#%%
    

import random

def mutate_normal(individual, mutation_rate, mutation_scale):
    """
    Perform mutation on an individual by adding noise from a normal distribution to each gene independently.
    
    Args:
        individual (list): The individual to mutate.
        mutation_rate (float): The probability of mutation for each gene.
        mutation_scale (float): The scale parameter of the normal distribution (standard deviation).
    
    Returns:
        list: The mutated individual.
    """
    mutated_individual = [] 
    for gene in individual:
        if random.random() < mutation_rate:
            mutation = random.gauss(0, mutation_scale)
            mutated_gene = gene + mutation
            mutated_individual.append(mutated_gene)
        else:
            mutated_individual.append(gene)
    return mutated_individual

# Example usage:
individual = [1.5, -2.7, 3.2, -4.1]
mutation_rate = 0.1
mutation_scale = 0.5

mutated_individual = mutate_normal(individual, mutation_rate, mutation_scale)
print("Original individual:", individual)
print("Mutated individual:", mutated_individual)
#%%
def genetic_algorithm(theta_hat):
    num_genes = len(theta_hat)
    population = gen_population(10, -3, 3, num_genes)

    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness
        
        #print(population)
        fitness_scores = [obj_func(individual) for individual in population]
        max_fitness = max(fitness_scores)
        print(f"Generation {generation}: Max Fitness = {max_fitness}")

        # Select parents and create offspring
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = get_parents(population)
            children = crossover_random(parent1, parent2, 3)
            print(parent1)
            print(parent2)
            print(children)
            new_population.extend(children)

        # Mutate offspring
        mutated_population = [mutate_normal(individual, mutation_rate, mutation_scale) for individual in new_population]

        # Replace old population with new population
        population = mutated_population

if __name__ == "__main__":
    genetic_algorithm([1, 3, 2])

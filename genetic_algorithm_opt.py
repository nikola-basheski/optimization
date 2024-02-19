#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:39:53 2024

@author: nikola
"""

import random

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
NUM_GENERATIONS = 100

# OneMax Problem Parameters
STRING_LENGTH = 20

def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = [random.randint(0, 1) for _ in range(STRING_LENGTH)]
        population.append(individual)
    return population

def calculate_fitness(individual):
    return sum(individual)

def select_parents(population):
    parent1 = random.choice(population)
    parent2 = random.choice(population)
    return parent1, parent2

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm():
    population = initialize_population()

    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness
        fitness_scores = [calculate_fitness(individual) for individual in population]
        max_fitness = max(fitness_scores)
        print(f"Generation {generation}: Max Fitness = {max_fitness}")

        # Select parents and create offspring
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])

        # Mutate offspring
        mutated_population = [mutate(individual) for individual in new_population]

        # Replace old population with new population
        population = mutated_population

if __name__ == "__main__":
    genetic_algorithm()

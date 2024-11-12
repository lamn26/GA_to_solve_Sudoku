import numpy as np, random
import matplotlib.pyplot as plt  
from utils import Utils  

# Class GA_Sudoku implements the Genetic Algorithm (GA) for solving Sudoku puzzles
class GA_Sudoku():
  
  # Initialization method for the class
  def __init__(self, sudoku, population_size = 2000, nums_of_generation = 500):
    # Initializes the Sudoku puzzle, population size, and number of generations for the GA
    self.sudoku = sudoku
    self.population_size = population_size  # Number of individuals in the population
    self.nums_of_generation = nums_of_generation  # Number of generations to evolve
    self.population = Utils.make_population(population_size, sudoku.full_grid)  # Creates the initial population
    self.num_to_reset = 100  # Number of generations before resetting the population if no improvement
    self.best_individual_overall = None  # Best individual across all generations

  # Natural selection process to select individuals for the next generation
  def natural_selection(self, elitism_size=0.1, rank_selection_size=0.4):
    # Sorts the population based on fitness (highest fitness first)
    self.population.sort(key=lambda ind: ind.fitness, reverse=True)

    # Elitism: Selects a portion of the population to carry over directly to the next generation
    num_elites = int(len(self.population) * elitism_size)
    elites = self.population[:num_elites]

    # Remaining individuals are ranked and selected based on fitness
    ranked_population = self.population[num_elites:]
    num_to_select = int(len(self.population) * rank_selection_size)

    # Normalizes the fitness values and computes selection probabilities
    fitness_values = np.array([ind.fitness for ind in ranked_population])
    fitness_values = fitness_values / np.median(fitness_values)
    exp_fitness_values = np.exp(fitness_values)
    probabilities = exp_fitness_values / np.sum(exp_fitness_values)

    # Select individuals based on their fitness probabilities
    selected_individuals = random.choices(
        ranked_population, weights=probabilities, k=num_to_select
    )

    # Updates the population with elites and selected individuals
    self.population = elites + selected_individuals

  # Main operation for running the GA to solve the Sudoku
  def operation(self):
    unchanged_generations = 0  # Counts generations with no improvement
    best_fitness = 0  # Tracks the best fitness found
    best_fitness_over_time = []  # List to store best fitness over time
    worst_fitness_over_time = []  # List to store worst fitness over time

    print("Finding solution ...")
    
    # Opens a file to store the progress of the GA
    with open("./sudoku_solution.txt", "w") as f:
      # Loop through each generation
      for i in range(self.nums_of_generation):
        # Gets the best and worst fitness in the current population
        current_best_individual = max(self.population, key=lambda ind: ind.fitness)
        current_best_fitness = current_best_individual.fitness
        current_worst_fitness = min(individual.fitness for individual in self.population)

        # Updates the best individual found so far
        if self.best_individual_overall is None or current_best_fitness > self.best_individual_overall.fitness:
          self.best_individual_overall = current_best_individual

        # Records the best and worst fitness over time
        best_fitness_over_time.append(current_best_fitness)
        worst_fitness_over_time.append(current_worst_fitness)

        # Checks if the fitness has improved, otherwise increments unchanged_generations
        if current_best_fitness != best_fitness:
          best_fitness = current_best_fitness
          unchanged_generations = 0
        else:
          unchanged_generations += 1

        # If solution is found (fitness = 243), exit the loop and save the solution
        if current_best_fitness == 243:
          print("Solution found!")
          f.write(f"{'-'*65}\nSolution found!\n")
          f.write(f"Generation {i}: Best fitness = {current_best_fitness}, Unchanged Generations: {unchanged_generations}\n")
          self.best_individual_overall.display_chromosome(file = f)
          print("Convergence plot saved as convergence_plot.png")
          self.plot_convergence(best_fitness_over_time, worst_fitness_over_time)
          return

        # Logs the current generation's best fitness and number of unchanged generations
        f.write(f"Generation {i}: Best fitness = {current_best_fitness}, Unchanged Generations: {unchanged_generations}\n")

        # If no improvement in fitness for a set number of generations, restart the population
        if unchanged_generations >= self.num_to_reset:
          f.write(f"{'-'*65}\nNo improvement in fitness, restarting population...\n")
          self.population = Utils.make_population(self.population_size, self.sudoku.full_grid)
          best_fitness = 0
          unchanged_generations = 0
          continue

        # Perform natural selection to update the population
        self.natural_selection()

        # Crossover and mutation to generate new offspring until population size is met
        new_population_size = len(self.population)
        while len(self.population) < self.population_size:
          parent_index_1 = random.randint(0, new_population_size - 1)
          parent_index_2 = random.randint(0, new_population_size - 1)
          while parent_index_1 == parent_index_2:
            parent_index_2 = random.randint(0, new_population_size - 1)

          # Select parents for crossover
          parent_1 = self.population[parent_index_1]
          parent_2 = self.population[parent_index_2]

          # Perform crossover and mutation to create two children
          child_1, child_2 = parent_1.crossover(parent_2)
          child_1.mutation()
          child_2.mutation()

          # Update fitness of the new children
          child_1.update_fitness()
          child_2.update_fitness()

          # Add the children to the population
          self.population.append(child_1)
          if len(self.population) < self.population_size:
            self.population.append(child_2)

    # If no solution is found after all generations, print the message and plot the convergence
    print("No solution found.")
    print("Convergence plot saved as convergence_plot.png")
    self.plot_convergence(best_fitness_over_time, worst_fitness_over_time)

  # Method to plot the convergence of fitness values over generations
  def plot_convergence(self, best_fitness, worst_fitness, filename="convergence_plot.png"):
    # Creates a plot showing the best and worst fitness over time
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness, label='Best Fitness')
    plt.plot(worst_fitness, label='Worst Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Convergence of GA Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as an image file
    plt.close()  # Close the plot


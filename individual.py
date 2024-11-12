import numpy as np
from sudoku import Sudoku
from utils import Utils

# Individual class representing a candidate solution (individual) in the genetic algorithm
class Individual:
    # Initialize an Individual with a Sudoku puzzle and a chromosome (solution representation)
    def __init__(self, sudoku, chromosome):
        self.sudoku = sudoku.make_copy()
        self.chromosome = chromosome
        self.fitness = 0  # Initialize the fitness value of the individual

    # Count the number of unique elements
    def count_unique_elements(self, line):
        return len(set(line))

    # Update the fitness of the individual based on its chromosome
    def update_fitness(self):
        self.fitness = 0  # Reset the fitness
        
        # Add the fitness from each row
        for row in self.chromosome:
            self.fitness += self.count_unique_elements(row)
        
        # Add the fitness from each column
        for col in self.chromosome.T:
            self.fitness += self.count_unique_elements(col)
        
        # Add the fitness from each 3x3 subgrid
        for i in range(3):
            for j in range(3):
                grid = self.chromosome[i * 3:i * 3 + 3, j * 3:j * 3 + 3].flatten()
                self.fitness += self.count_unique_elements(grid)

    # Perform crossover between two individuals at a given point
    def crossover(self, other_individual, crossover_point=0.6):
        child_1, child_2 = [], []  # Initialize the chromosomes for two offspring
        for i in range(9):  
            new_row_child_1, new_row_child_2 = [], []  # New rows for the children
            for j in range(9):
                # Perform crossover with a probability of crossover_point
                if np.random.rand() < crossover_point:
                    new_row_child_1.append(other_individual.chromosome[i][j])
                    new_row_child_2.append(self.chromosome[i][j])
                else:
                    new_row_child_1.append(self.chromosome[i][j])
                    new_row_child_2.append(other_individual.chromosome[i][j])
            child_1.append(new_row_child_1)  # Add the row to the first child
            child_2.append(new_row_child_2)  # Add the row to the second child
        
        # Return two new Individual objects (offspring) created by the crossover
        individual_1, individual_2 = Individual(Sudoku(init = np.array(child_1)), np.array(child_1)), Individual(Sudoku(init = np.array(child_2)), np.array(child_2))

        individual_1.update_fitness()
        individual_2.update_fitness()

        return individual_1, individual_2
        
    # Perform mutation on the individual's chromosome with a given mutation rate
    def mutation(self, mutation_rate=0.05):
        for i in range(9):  # Iterate over each row
            if np.random.rand() < mutation_rate:  # With probability mutation_rate
                self.chromosome[i] = Utils.make_gene(self.sudoku.full_grid[i])  # Generate a new gene for the row
        # After mutation, update the individual's fitness
        self.update_fitness()
        
    # Return the fitness of the individual
    def fitness_calculation(self):
        return self.fitness

    # Display the individual's chromosome
    def display_chromosome(self, file=None): 
        output_lines = []

        for i, row in enumerate(self.chromosome):
            line = " ".join(str(cell) for cell in row[:3]) + " | " + \
                   " ".join(str(cell) for cell in row[3:6]) + " | " + \
                   " ".join(str(cell) for cell in row[6:])
            output_lines.append(line)
            print(line)
            
            if file:
                file.write(line + '\n')

            if i % 3 == 2 and i != 8:
                separator = "-" * 21
                output_lines.append(separator)
                print(separator)
                if file:
                    file.write(separator + '\n')

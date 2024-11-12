import numpy as np
import random
import matplotlib.pyplot as plt

class Utils():
  @staticmethod
  def make_gene(initial=None):
    if initial is None:
      initial = [0] * 9

    gene = [0] * 9
    fixed_values = set()

    for i in range(9):
      if initial[i] != 0:
        gene[i] = initial[i]
        fixed_values.add(initial[i])

    remaining_values = [num for num in range(1, 10) if num not in fixed_values]
    random.shuffle(remaining_values)

    for i in range(9):
      if gene[i] == 0:
        gene[i] = remaining_values.pop()

    return np.array(gene)

  @staticmethod
  def make_chromosome(initial=None):
    if initial is None:
      initial = [[0]*9]*9
    chromosome = []
    for i in range(9):
      chromosome.append(Utils.make_gene(initial[i]))
    return np.array(chromosome)

  @staticmethod
  def make_population(nums, initial=None):
    population = []
    for _ in range(nums):
      initial_sudoku = Sudoku(init = initial)
      initial_chromosome = Utils.make_chromosome(initial)
      individual = Individual(initial_sudoku, initial_chromosome)
      individual.update_fitness()
      population.append(individual)
    return population
  
  @staticmethod
  def is_valid(board, row, col, num):
    for i in range(9):
      if board[row][i] == num or board[i][col] == num:
        return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
      for j in range(3):
        if board[start_row + i][start_col + j] == num:
          return False
    return True

  @staticmethod
  def fill_board(board):
    for i in range(9):
      for j in range(9):
        if board[i][j] == 0:
          numbers = list(range(1, 10))
          random.shuffle(numbers)
          for num in numbers:
            if Utils.is_valid(board, i, j, num):
              board[i][j] = num
              if Utils.fill_board(board):
                return True
              board[i][j] = 0
          return False
    return True

  @staticmethod
  def create_sudoku(num_known_cells, filename="./sudoku.txt"):
    board = [[0] * 9 for _ in range(9)]
    Utils.fill_board(board)

    with open("./final_sudoku.txt", "w") as f:
      for row in board:
        f.write(", ".join(map(str, row)) + "\n")

    cells = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(cells)
    for i, j in cells[num_known_cells:]:
      board[i][j] = 0

    with open(filename, "w") as f:
      for row in board:
        f.write(", ".join(map(str, row)) + "\n")

    return board

class Sudoku():
  def __init__(self, nums_of_known_cells = 42, init = None):
    if init is not None:
      self.full_grid = init
    else:
      self.full_grid = Utils.create_sudoku(nums_of_known_cells, "sudoku.txt")

    self.full_grid = np.array(self.full_grid)

  def check(self):
    for row in self.full_grid:
      numbers = [num for num in row if num != 0]
      if len(set(numbers)) != len(numbers):
        return False

    for col in self.full_grid.T:
      numbers = [num for num in col if num != 0]
      if len(set(numbers)) != len(numbers):
        return False

    for i in range(3):
      for j in range(3):
        grid = self.full_grid[i*3:i*3+3, j*3:j*3+3]
        numbers = [num for row in grid for num in row if num != 0]
        if len(set(numbers)) != len(numbers):
          return False

    return True

  def make_copy(self):
    return Sudoku(init = self.full_grid.copy())

  def display(self):
    for i, row in enumerate(self.full_grid):
      print(" ".join(str(cell) for cell in row[:3]), "|",
            " ".join(str(cell) for cell in row[3:6]), "|",
            " ".join(str(cell) for cell in row[6:]))

      if i % 3 == 2 and i != 8:
          print("-" * 21)

class Individual():
  def __init__(self, sudoku, chromosome):
    self.sudoku = sudoku.make_copy()
    self.chromosome = chromosome
    self.fitness = 0

  def count_unique_elements(self, line):
    return len(set(line))

  def update_fitness(self):
    self.fitness = 0

    # row
    for row in self.chromosome:
      self.fitness += self.count_unique_elements(row)

    # col
    for col in self.chromosome.T:
      self.fitness += self.count_unique_elements(col)

    # grid
    for i in range(3):
      for j in range(3):
        grid = self.chromosome[i*3:i*3+3, j*3:j*3+3].flatten()
        self.fitness += self.count_unique_elements(grid)

  def crossover(self, other_individual, crossover_point = 0.6):
    child_1 = [] # other
    child_2 = [] # this

    for i in range(9):
      new_row_child_1 = []
      new_row_child_2 = []
      for j in range(9):
        if random.random() < crossover_point:
          new_row_child_1.append(other_individual.chromosome[i][j])
          new_row_child_2.append(self.chromosome[i][j])
        else:
          new_row_child_1.append(self.chromosome[i][j])
          new_row_child_2.append(other_individual.chromosome[i][j])
      child_1.append(new_row_child_1)
      child_2.append(new_row_child_2)

    individual_1, individual_2 = Individual(Sudoku(init = np.array(child_1)), np.array(child_1)), Individual(Sudoku(init = np.array(child_2)), np.array(child_2))

    individual_1.update_fitness()
    individual_2.update_fitness()

    return individual_1, individual_2

  def mutation(self, mutation_rate = 0.05):
    for i in range(9):
      if random.random() < mutation_rate:
        self.chromosome[i] = Utils.make_gene(self.sudoku.full_grid[i])

    self.update_fitness()

  def fitness_calculation(self):
    return self.fitness

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

class GA_Sudoku():
  def __init__(self, sudoku, population_size = 2000, nums_of_generation = 500):
    self.sudoku = sudoku
    self.population_size = population_size
    self.nums_of_generation = nums_of_generation
    self.population = Utils.make_population(population_size, sudoku.full_grid)
    self.num_to_reset = 100
    self.best_individual_overall = None


  def natural_selection(self, elitism_size=0.1, rank_selection_size=0.4):
    self.population.sort(key=lambda ind: ind.fitness, reverse=True)

    num_elites = int(len(self.population) * elitism_size)
    elites = self.population[:num_elites]

    ranked_population = self.population[num_elites:]
    num_to_select = int(len(self.population) * rank_selection_size)

    fitness_values = np.array([ind.fitness for ind in ranked_population])
    fitness_values = fitness_values / np.median(fitness_values)

    exp_fitness_values = np.exp(fitness_values)
    probabilities = exp_fitness_values / np.sum(exp_fitness_values)

    selected_individuals = random.choices(
        ranked_population, weights=probabilities, k=num_to_select
    )

    self.population = elites + selected_individuals

  def operation(self):
    unchanged_generations = 0
    best_fitness = 0
    best_fitness_over_time = []
    worst_fitness_over_time = []

    print("Finding solution ...")
    
    with open("./sudoku_solution.txt", "w") as f:
      for i in range(self.nums_of_generation):
        current_best_individual = max(self.population, key=lambda ind: ind.fitness)
        current_best_fitness = current_best_individual.fitness
        current_worst_fitness = min(individual.fitness for individual in self.population)

        if self.best_individual_overall is None or current_best_fitness > self.best_individual_overall.fitness:
          self.best_individual_overall = current_best_individual

        best_fitness_over_time.append(current_best_fitness)
        worst_fitness_over_time.append(current_worst_fitness)

        if current_best_fitness != best_fitness:
          best_fitness = current_best_fitness
          unchanged_generations = 0
        else:
          unchanged_generations += 1

        if current_best_fitness == 243:
          print("Solution found!")
          f.write(f"{'-'*65}\nSolution found!\n")
          f.write(f"Generation {i}: Best fitness = {current_best_fitness}, Unchanged Generations: {unchanged_generations}\n")
          self.best_individual_overall.display_chromosome(file = f)
          print("Convergence plot saved as convergence_plot.png")
          self.plot_convergence(best_fitness_over_time, worst_fitness_over_time)
          return

        f.write(f"Generation {i}: Best fitness = {current_best_fitness}, Unchanged Generations: {unchanged_generations}\n")

        if unchanged_generations >= self.num_to_reset:
          f.write(f"{'-'*65}\nNo improvement in fitness, restarting population...\n")
          self.population = Utils.make_population(self.population_size, self.sudoku.full_grid)
          best_fitness = 0
          unchanged_generations = 0
          continue

        # natural selection
        self.natural_selection()
        new_population_size = len(self.population)

        # crossover and mutate
        while len(self.population) < self.population_size:
          parent_index_1 = random.randint(0, new_population_size - 1)
          parent_index_2 = random.randint(0, new_population_size - 1)
          while parent_index_1 == parent_index_2:
            parent_index_2 = random.randint(0, new_population_size - 1)

          parent_1 = self.population[parent_index_1]
          parent_2 = self.population[parent_index_2]

          child_1, child_2 = parent_1.crossover(parent_2)
          child_1.mutation()
          child_2.mutation()

          child_1.update_fitness()
          child_2.update_fitness()

          self.population.append(child_1)
          if len(self.population) < self.population_size:
            self.population.append(child_2)

    print("No solution found.")
    print("Convergence plot saved as convergence_plot.png")
    self.plot_convergence(best_fitness_over_time, worst_fitness_over_time)

  def plot_convergence(self, best_fitness, worst_fitness, filename="convergence_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness, label='Best Fitness')
    plt.plot(worst_fitness, label='Worst Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Convergence of GA Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
  sudoku = Sudoku()
  population = GA_Sudoku(sudoku=sudoku, nums_of_generation=1000)
  population.operation()
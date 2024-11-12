import numpy as np
import random

class Utils:
    # Static method to generate a gene (a row of the Sudoku board)
    @staticmethod
    def make_gene(initial=None):
        if initial is None:
            initial = [0] * 9  # Default gene with 9 zeros

        gene = [0] * 9
        fixed_values = set()

        # Initialize the gene with fixed values from the 'initial' input
        for i in range(9):
            if initial[i] != 0:
                gene[i] = initial[i]
                fixed_values.add(initial[i])

        # Get the remaining values that are not in the 'fixed_values' set
        remaining_values = [num for num in range(1, 10) if num not in fixed_values]
        random.shuffle(remaining_values)

        # Fill the empty positions in the gene with the remaining values
        for i in range(9):
            if gene[i] == 0:
                gene[i] = remaining_values.pop()

        return np.array(gene)

    # Static method to create a chromosome (a full Sudoku grid)
    @staticmethod
    def make_chromosome(initial=None):
        if initial is None:
            initial = [[0]*9]*9  # Default empty grid (9x9)

        chromosome = []
        for i in range(9):
            chromosome.append(Utils.make_gene(initial[i]))  # Create each gene (row)
        return np.array(chromosome)

    # Static method to create a population of individuals (Sudoku solutions)
    @staticmethod
    def make_population(nums, initial=None):
        from sudoku import Sudoku
        from individual import Individual
        population = []
        for _ in range(nums):
            initial_sudoku = Sudoku(init=initial)
            initial_chromosome = Utils.make_chromosome(initial)
            individual = Individual(initial_sudoku, initial_chromosome)
            individual.update_fitness()  # Calculate fitness of the individual
            population.append(individual)
        return population
    
    # Static method to check if a number can be placed in a cell of the board
    @staticmethod
    def is_valid(board, row, col, num):
        # Check row and column for duplicate numbers
        for i in range(9):
            if board[row][i] == num or board[i][col] == num:
                return False
        # Check the 3x3 subgrid for duplicates
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False
        return True

    # Static method to recursively fill the Sudoku board with valid numbers
    @staticmethod
    def fill_board(board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    numbers = list(range(1, 10))  # Try numbers from 1 to 9
                    random.shuffle(numbers)
                    for num in numbers:
                        if Utils.is_valid(board, i, j, num):  # Check if the number is valid
                            board[i][j] = num
                            if Utils.fill_board(board):  # Recursively fill the board
                                return True
                            board[i][j] = 0  # Backtrack if no valid number found
                    return False
        return True

    # Static method to create a Sudoku puzzle with a given number of known cells
    @staticmethod
    def create_sudoku(num_known_cells, filename="./sudoku.txt"):
        board = [[0] * 9 for _ in range(9)]
        Utils.fill_board(board)  # Fill the board with a valid solution

        # Save the filled Sudoku board to a file
        with open("./final_sudoku.txt", "w") as f:
            for row in board:
                f.write(", ".join(map(str, row)) + "\n")

        # Remove some cells to create the puzzle
        cells = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(cells)
        for i, j in cells[num_known_cells:]:
            board[i][j] = 0  # Set some cells to 0 to create the puzzle

        # Save the puzzle to a file
        with open(filename, "w") as f:
            for row in board:
                f.write(", ".join(map(str, row)) + "\n")

        return board

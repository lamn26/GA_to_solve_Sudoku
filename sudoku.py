import numpy as np
from utils import Utils

# Sudoku class representing a Sudoku puzzle
class Sudoku:
    # Initialize a Sudoku puzzle
    def __init__(self, nums_of_known_cells=45, init=None):
        # If an initial grid is provided, use it; otherwise, create a new Sudoku puzzle
        if init is not None:
            self.full_grid = init
        else:
            # Generate a new Sudoku puzzle with a specified number of known cells
            self.full_grid = Utils.create_sudoku(nums_of_known_cells)
        
        # Convert the grid to a numpy array for easier manipulation
        self.full_grid = np.array(self.full_grid)

    # Create and return a copy of the current Sudoku object
    def make_copy(self):
        return Sudoku(init=self.full_grid.copy())  # Use np.copy() to avoid reference issues

    # Display the Sudoku grid in a human-readable format
    def display(self):
        for i, row in enumerate(self.full_grid):
            print(" ".join(str(cell) for cell in row[:3]), "|",
                  " ".join(str(cell) for cell in row[3:6]), "|",
                  " ".join(str(cell) for cell in row[6:]))
            
            if i % 3 == 2 and i != 8:
                print("-" * 21)

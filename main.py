from sudoku import Sudoku
from ga_sudoku import GA_Sudoku

if __name__ == "__main__":
    # Initialize the Sudoku puzzle with 45 known cells
    sudoku = Sudoku(nums_of_known_cells=45)  # nums_of_known_cells must be larger than 0 and smaller than 81. Suggested 81 >= nums_of_known_cells >= 45
    
    # Create the GA_Sudoku population with 1000 generations
    population = GA_Sudoku(sudoku=sudoku, nums_of_generation=1000)
    
    # Run the GA operation to solve the Sudoku
    population.operation()
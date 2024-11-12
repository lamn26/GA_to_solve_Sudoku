# Genetic algorithm to solve Sudoku

This project utilizes a genetic algorithm to solve Sudoku puzzles by simulating the process of natural evolution. The algorithm generates an initial population of random Sudoku grids, then iteratively selects the best solutions based on fitness criteria, such as minimizing the number of conflicts in rows, columns, and subgrids. Through crossover, mutation, and selection, the population evolves towards the optimal solution. Over successive generations, the algorithm converges to a valid Sudoku grid that satisfies all the puzzle's constraints.

## Install

1. Clone the repository:
    ```bash
    git clone https://github.com/lamn26/GA_to_solve_Sudoku.git
    cd GA_to_solve_Sudoku
    ```

2. Install all required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Run

```bash
python main.py
```

## Output Structure

1. **sudoku.txt**: Sudoku initial state.
2. **final_sudoku.txt**: Sudoku ending state.
3. **sudoku_solution.txt**: Sudoku solving process (describe the best fitness through each step)
4. **convergence_plot.png**: Convergence graph
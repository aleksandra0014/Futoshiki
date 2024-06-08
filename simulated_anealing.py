import random
import math
import numpy as np
from matplotlib import pyplot as plt

DEFAULT_BOARD_SIZE = 5
constraint1 = [[(0, 0), (0, 1), ">"],
              [(1, 3), (1, 4), "<"],
              [(3, 1), (3, 2), "<"]]


class SimulatedAnnealing:

    def __init__(self,  constraints, board_size=DEFAULT_BOARD_SIZE):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))
        self.constraints = constraints

    def random_matrix(self):
        return np.random.randint(1, self.board_size + 1, size=(self.board_size, self.board_size))

    @staticmethod
    def count_repeated_elements(row):
        count = 0
        element_counts = {}
        for element in row:
            if element in element_counts:
                element_counts[element] += 1
            else:
                element_counts[element] = 1

        for i in element_counts.values():
            if i >= 2:
                count += i-1
        return count

    def check_constraints(self, matrix):
        error = 0
        for c in self.constraints:
            x1, y1 = c[0]
            x2, y2 = c[1]
            if c[2] == '>':
                if matrix[x1, y1] > matrix[x2, y2]:
                    pass
                else:
                    error += 1
            elif c[2] == '<':
                if matrix[x1, y1] < matrix[x2, y2]:
                    pass
                else:
                    error += 1
        return error

    def cost(self, matrix):
        # sumować liczbe powtórzen liczb i dodawać +1 za kazde nie spelnione ograniczenie
        cost = 0
        for row in matrix:
            cost += self.count_repeated_elements(row)
        for col in matrix.T:
            cost += self.count_repeated_elements(col)
        cost += self.check_constraints(matrix)
        return cost

    @staticmethod
    def generate_new_matrix(matrix):
        matrix = matrix.copy()
        num_rows = len(matrix)
        num_cols = len(matrix[0])
        row = random.randint(0, num_rows - 1)
        col = random.randint(0, num_cols - 1)

        i = random.randint(0, num_cols - 1)
        j = random.randint(0, num_cols - 1)
        while i == j:
            i = random.randint(0, num_cols - 1)
            j = random.randint(0, num_cols - 1)
        matrix[row][i], matrix[row][j] = matrix[row][j], matrix[row][i]
        matrix[i][col], matrix[j][col] = matrix[j][col], matrix[i][col]
        return matrix

    def simulated_annealing(self, temperature=10.0, cooling_rate=0.98, max_iterations=10000):

        matrix = self.random_matrix()
        best_matrix = matrix
        best_cost = self.cost(matrix)

        iterations = 0

        while temperature > 0.1 and iterations < max_iterations:
            new_matrix = self.generate_new_matrix(matrix)
            new_cost = self.cost(new_matrix)

            # SPRAWDZANIE WARUNKÓW
            if new_cost < best_cost:
                best_matrix = new_matrix
                best_cost = new_cost

            if new_cost < self.cost(matrix):
                matrix = new_matrix
            elif random.random() < pow(
                    math.e, (self.cost(matrix) - new_cost) / temperature
            ):
                matrix = new_matrix

            temperature *= cooling_rate
            iterations += 1

        return best_matrix, best_cost, iterations

    def plot_statistic(self, stat="median", type="cooling rate", repeat_iterations=100, temperature=10,
                       cooling_rate=0.99, max_iterations=1000):

        if stat == "median":
            statistic_function = np.median
        elif stat == "mean":
            statistic_function = np.mean
        else:
            raise ValueError("Unsupported statistic type. Use 'mean' or 'median'.")

        if type == "cooling rate":
            stat_cost_per_cr = []
            cooling_rates = np.arange(0.8, 0.99, 0.01)

            for cr in cooling_rates:
                costs = []
                for i in range(repeat_iterations):
                    _, best_cost, _ = self.simulated_annealing(temperature, cr, max_iterations)
                    costs.append(best_cost)
                stat_cost_per_cr.append(statistic_function(costs))

            self.plot_costs(cooling_rates, stat_cost_per_cr, type, stat.capitalize())

        elif type == "temperature":
            stat_cost_per_temp = []
            temperatures = np.arange(1, 10, 0.5)

            for temp in temperatures:
                costs = []
                for i in range(repeat_iterations):
                    _, best_cost, _ = self.simulated_annealing(temp, cooling_rate, max_iterations)
                    costs.append(best_cost)
                stat_cost_per_temp.append(statistic_function(costs))

            self.plot_costs(temperatures, stat_cost_per_temp, type, stat.capitalize())

        else:
            raise ValueError("Unsupported type. Use 'cooling_rate' or 'temperature'.")

    @staticmethod
    def plot_costs(x, y, type, title):
        plt.style.use('ggplot')
        plt.plot(x, y)
        plt.xlabel(type)
        plt.ylabel(f"{title} costs")
        plt.title(f"{title} cost for different {type}")
        plt.grid(True)
        plt.show()



if __name__ == "__main__":
    s = SimulatedAnnealing(constraint1)
    best_matrix, best_cost, iterations = s.simulated_annealing()
    print("Best matrix:")
    print(best_matrix)
    print("Best cost:", best_cost)
    print("Iterations:", iterations)
    s.plot_statistic(type='cooling rate')
    s.plot_statistic(type='temperature')
    s.plot_statistic(type='cooling rate', stat='mean')
    s.plot_statistic(type='temperature', stat='mean')
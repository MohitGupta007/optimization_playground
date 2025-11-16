import numpy as np

class RunTracker:
    def __init__(self, csv_path=None, pop_size=None, generations=None):
        self.pop_size = pop_size
        self.generations = generations
        self.csv_path = csv_path
        if self.csv_path:
            # clear existing file
            with open(self.csv_path, 'w') as f_csv:
                pass

    def log_evaluation(self, x, f):
        if self.csv_path:
            # append to CSV file
            with open(self.csv_path, 'a') as f_csv:
                if(len(x) == 1):
                    f_csv.write(f"{str(x[0])},{f}\n")
                else:
                    f_csv.write(f"{','.join(map(str, x))},{f}\n")

    def get_history(self):
        if self.csv_path:
            data = np.loadtxt(self.csv_path, delimiter=',')
            X_history = data[:, :-1]
            F_history = data[:, -1]
            return X_history, F_history
        else:
            return None, None

    def set_algo_name(self, algo_name):
        self.algo_name = algo_name

    def set_problem_name(self, problem_name):
        self.problem_name = problem_name

    def set_bounds(self, bounds):
        self.bounds = np.array(list(bounds))
from sklearn.neural_network import MLPRegressor
from utils import simulate_rule, read_from_file

def main() -> None:
    coords = read_from_file()

    rule1 = MLPRegressor(hidden_layer_sizes=(50, 50, 50, 50),
                         max_iter=5000,
                         activation='tanh',
                         solver='adam',
                         learning_rate='adaptive',
                         n_iter_no_change=1000)

    rule2= MLPRegressor(hidden_layer_sizes=(4, 4, 4),
                        max_iter=20000,
                        activation='logistic',
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=10000)

    rule3 = MLPRegressor(hidden_layer_sizes=(5, 10, 20, 30, 40, 50),
                         max_iter=5000,
                         activation='tanh',
                         solver='adam',
                         learning_rate='adaptive',
                         n_iter_no_change=1000)

    simulate_rule(rule1, coords)
    simulate_rule(rule2, coords)
    simulate_rule(rule3, coords)

main()
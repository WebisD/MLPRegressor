from sklearn.neural_network import MLPRegressor
from utils import simulate_rule, read_from_file

def main() -> None:
    coords = read_from_file()

    rule1 = MLPRegressor(hidden_layer_sizes=(10,10),
                        max_iter=1000,
                        activation='relu',
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=100)

    rule2= MLPRegressor(hidden_layer_sizes=(2),
                        max_iter=20000,
                        activation='relu',
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=2000)

    rule3 = MLPRegressor(hidden_layer_sizes=(5,10,20,30),
                         max_iter=1000,
                         activation='relu',
                         solver='adam',
                         learning_rate='adaptive',
                         n_iter_no_change=100)

    simulate_rule(rule1, coords)
    simulate_rule(rule2, coords)
    simulate_rule(rule3, coords)

main()
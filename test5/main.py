from sklearn.neural_network import MLPRegressor
from utils import simulate_rule, read_from_file

def main() -> None:
    coords = read_from_file()

    rule1 = MLPRegressor(hidden_layer_sizes=(40, 50, 60, 100),
                         max_iter=65000,
                         activation='logistic',
                         solver='adam',
                         learning_rate='adaptive',
                         n_iter_no_change=7000)

    # rule2= MLPRegressor(hidden_layer_sizes=(150),
    #                     max_iter=20000,
    #                     activation='relu',
    #                     solver='adam',
    #                     learning_rate = 'adaptive',
    #                     n_iter_no_change=5500)
    #
    # rule3 = MLPRegressor(hidden_layer_sizes=(150, 50, 50, 150),
    #                      max_iter=1000,
    #                      activation='relu',
    #                      solver='adam',
    #                      learning_rate='adaptive',
    #                      n_iter_no_change=500)

    simulate_rule(rule1, coords)
    # simulate_rule(rule2, coords)
    # simulate_rule(rule3, coords)

main()

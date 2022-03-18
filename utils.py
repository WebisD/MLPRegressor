import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def read_from_file() -> tuple[float, float]:
    print('Loading test file')

    file = np.load(f'dataset.npy')
    x = file[0]
    y = np.ravel(file[1])

    return (x, y)

def plot_graph(rule: MLPRegressor, y_est: np.ndarray, coords: tuple[float, float]) -> None:
    x, y = coords
    plt.figure(figsize=[14,7])

    #plot curso original
    plt.subplot(1,3,1)
    plt.plot(x,y)

    #plot aprendizagem
    plt.subplot(1,3,2)
    plt.plot(rule.loss_curve_)

    #plot regressor
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='yellow')
    plt.plot(x,y_est,linewidth=2)


    plt.show()

def fit_neural(rule: MLPRegressor, coords: tuple[float, float]) -> None:
    error = []
    x, y = coords

    for i in range(5):
        rule = rule.fit(x,y)
        error.append(rule.loss_curve_[-1])

    print("Média do erro:", np.mean(error))
    print("Desvio padrão do erro:",np.std(error))

def simulate_rule(rule: MLPRegressor, coords: tuple[float, float]) -> None:
    x, y = coords
    print("Training")
    rule = rule.fit(x, y)

    plot_graph(rule, rule.predict(x), coords)
    fit_neural(rule, coords)

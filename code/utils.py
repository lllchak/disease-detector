import matplotlib.pyplot as plt


def plot_metrics(history: dict) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(history['train'])
    plt.plot(history['val'])
    plt.legend(list(history.keys()))

    plt.show()

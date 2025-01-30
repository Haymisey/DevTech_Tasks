import matplotlib.pyplot as plt
import seaborn as sns

def plot_precision_at_k(precision_values):
    plt.plot(range(1, len(precision_values) + 1), precision_values)
    plt.xlabel('K')
    plt.ylabel('Precision')
    plt.title('Precision at K')
    plt.show()

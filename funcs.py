import matplotlib.pyplot as plt
import numpy as np
def plot_data(x,y):
    plt.scatter(x, y)
    plt.xlabel(u"x")
    plt.ylabel(u"y")

def plot_line(w,b):
    x_ = np.linspace(0, 1, 2)
    y_draw = w * x_ + b
    plt.plot(x_, y_draw,color='green')
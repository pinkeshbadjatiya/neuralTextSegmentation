from matplotlib import pyplot as plt

MARKERS = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
COLORS = ('blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white')


def plot(X, Y, axis_prefix=""):
    ax = plt.subplot(111)
                      
    for i in range(len(X)):
        plt.scatter(x=X[i],
                y=Y[i],
                marker=MARKERS[0],
                color=COLORS[0],
                alpha=0.5,
                )
    plt.xlabel(axis_prefix + 'Iteration')
    plt.ylabel(axis_prefix + 'Error')
                      
    #leg = plt.legend(loc='upper right', fancybox=True)
    #leg.get_frame().set_alpha(0.5)
    plt.title('%s: Distribution of X vs Y' %(axis_prefix))
                      
    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")
                      
    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
                      
    plt.grid()
    plt.tight_layout
    plt.show()

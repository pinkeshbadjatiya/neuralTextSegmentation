from matplotlib import pyplot as plt

MARKERS = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')
COLORS = ('blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white')


def plot(X, Y, axis_prefix="Axis"):
    ax = plt.subplot(111)
                      
    class_markers, class_colors = MARKERS[:len(classes)], COLORS[:len(classes)]
    for i in range(len(X)):
        plt.scatter(x=X[i],
                y=Y[i],
                marker=MARKERS[0],
                color=COLORS[0],
                alpha=0.5,
                label='Error per iteration'
                )
    plt.xlabel(axis_prefix + '1')
    plt.ylabel(axis_prefix + '2')
                      
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
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

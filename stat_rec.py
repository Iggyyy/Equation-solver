test_accuracies = []

preds_vs_actual = []


def add_acc(x):
    test_accuracies.append(x)
def add_results(pred, label):
    preds_vs_actual.append([pred, label])
def stat_show():

    import matplotlib.pyplot as plt

    iter_step = 5

    i_axis = list(range(0,len(test_accuracies)*iter_step, iter_step))
    #print(len(test_accuracies), i_axis, test_accuracies)

    plt.plot(i_axis, test_accuracies)
    plt.title("Test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.show()
    


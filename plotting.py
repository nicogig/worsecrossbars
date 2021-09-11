import numpy as np
import matplotlib.pyplot as plt

def training_validation_plotter (epochs, training, validation, value_type="", fpath=None, filename="", label_step=1):
    """
    training_validation_plotter:
    Plot the Training and Validation curves with respect to the Epochs.
    Inputs:
        -   epochs: The range of the epochs used to train the network.
        -   training: An array (list) containing the values obtained from the training stage.
        -   validation: An array (list) containing the values obtained from the validation stage.
        -   value_type: A string describing the type of data. Allowed values are "Accuracy" and "Loss". 
        -   fpath: The FontProperties object containing information about the stop to use.
    Optional Input:
        -   label_step: The step for the labels on the x-axis. Defaults to 1.
    Output:
        -   The graphs, both inline and as .png files.
    """

    if (value_type.lower() != "accuracy" and value_type.lower() != "loss"):
        raise ValueError('"value_type" parameter must be either "Accuracy" or "Loss".')
    
    if fpath == None:
        raise ValueError('"fpath" parameter must be passed to the plotting function.')
    

    fig = plt.figure()
    fig.set_size_inches(10, 6)

    title = "Training and validation " + value_type.lower()
    y_label = value_type.title()
    training_legend_label = "Training " + value_type.lower()
    validation_legend_label = "Validation " + value_type.lower()
    
    plt.plot(epochs, training, '-bo', markersize=7, label=training_legend_label)
    plt.plot(epochs, validation, '-rD', markersize=7, label=validation_legend_label)

    plt.xlabel("Epochs", font=fpath, fontsize=20)
    plt.ylabel(y_label, font=fpath, fontsize=20)
    plt.grid()
    plt.tight_layout()
    plt.xticks(np.arange(0, len(epochs) + 1, step=label_step), font=fpath, fontsize=15)
    plt.yticks(font=fpath, fontsize=15)

    if value_type.lower() == "accuracy":
        L = plt.legend(fontsize=15, loc='lower right')
    elif value_type.lower() == "loss":
        L = plt.legend(fontsize=15, loc='upper right')
    else:
        L = plt.legend(fontsize=15)
    
    plt.setp(L.texts, font=fpath)

    if filename != "":
        plt.savefig(filename + ".png")
    
    plt.title(title, font=fpath, fontsize=20)
    plt.show()
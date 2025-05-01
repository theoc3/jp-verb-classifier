import pandas as pd
import matplotlib.pyplot as plt

def bilstm_graph():
    # Load the data
    data = pd.read_csv('models/bilstm_epochs.csv')

    # Plot Training and Evaluation Loss
    plt.figure()
    plt.plot(data['Epoch'], data['Train Loss'], label='Train Loss')
    plt.plot(data['Epoch'], data['Eval Loss'], label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_plot.png')
    plt.show()

    # Plot Training and Evaluation Accuracy
    plt.figure()
    plt.plot(data['Epoch'], data['Train Acc (%)'], label='Train Accuracy')
    plt.plot(data['Epoch'], data['Eval Acc (%)'], label='Eval Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_plot.png')
    plt.show()
    
bilstm_graph()
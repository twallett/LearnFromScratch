#%%
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_error(error):
    plt.plot(error)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("MLP Classifier Cross-entropy")
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(os.path.join('plots', "MLPClassifier_crossentropy.png"))
    plt.show()
    
def animate(X_test, y_test, predictions):

    fig, ax = plt.subplots()
    
    def animate(i):
        ax.clear()
        ax.imshow(X_test[i])
        ax.set_title(f"Prediction: {predictions[i]} - Ground Truth: {int(y_test[i])}")
        ax.axis('off')

    ani = FuncAnimation(fig, animate, frames=50, interval=1000)
    if not os.path.exists('plots'):
        os.makedirs('plots')
    ani.save(os.path.join('plots', "MLPClassifier_anim.gif"), writer='ffmpeg')

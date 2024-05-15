import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

def plot_sse(error, xor = None):
    plt.plot(error ** 2)
    plt.title("Perceptron SSE")
    plt.xlabel("Iterations")
    plt.ylabel("SSE")
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if xor:
        plt.savefig(os.path.join('plots', 'perceptron_sse_xor.png'))
    else:
        plt.savefig(os.path.join('plots', 'perceptron_sse.png'))
    plt.show()
    
def animate(inputs, targets, param, xor = None):
    
    #Perceptron decision boundary
    p1 = inputs[:,0]
    p2 = inputs[:,1]

    p1_label_0 = [p1[enum] for enum, i in enumerate(targets) if i == 0]
    p1_label_1 = [p1[enum] for enum, i in enumerate(targets) if i == 1]
    p2_label_0 = [p2[enum] for enum, i in enumerate(targets) if i == 0]
    p2_label_1 = [p2[enum] for enum, i in enumerate(targets) if i == 1]

    fig, ax = plt.subplots()

    def anim(i):
        
        quiver_1 = ax.quiver(0, 0, 0, 0)
        
        ax.clear()
        
        plt.scatter(p1_label_0, p2_label_0, label='Class 0')
        plt.scatter(p1_label_1, p2_label_1, c='orange', label= 'Class 1')
        
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        
        p1_w = float(param[i][0][0][0])
        p2_w = float(param[i][0][0][1])

        if p2_w == 0:
            db_slope = 0
        else:
            db_slope = (-p1_w)/p2_w
        db_intercept = (-param[i][1])/p2_w

        db_1 = db_slope + db_intercept
        
        quiver_1.set_UVC(np.array(p1_w), np.array([p2_w]))
        ax.add_artist(quiver_1)
        ax.axline((1, db_1), slope=db_slope)
        
        plt.title("Perceptron")
        plt.xlabel("$P1$")
        plt.ylabel("$P2$")
        plt.legend(loc = 'upper right')
        
        return quiver_1, 

    anim = FuncAnimation(fig, anim, frames=len(inputs))
       
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if xor:
        anim.save(os.path.join('plots', 'perceptron_anim_xor.gif'), writer='ffmpeg', fps =10)
    else:
        anim.save(os.path.join('plots', 'perceptron_anim.gif'), writer='ffmpeg', fps =10)
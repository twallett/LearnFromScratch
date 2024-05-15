import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

def plot_sse(error):
    plt.plot(error ** 2)
    plt.title("ADALINE SSE")
    plt.xlabel("Iterations")
    plt.ylabel("SSE")
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(os.path.join('plots', 'adaline_sse.png'))
    plt.show()
    
def animate(inputs, targets, param):
    
    class1 = [inputs[:, i] for i in range(inputs.shape[1]) if targets[0, i] == -1 and targets[1, i] == -1]
    class2 = [inputs[:, i] for i in range(inputs.shape[1]) if targets[0, i] == -1 and targets[1, i] == 1]
    class3 = [inputs[:, i] for i in range(inputs.shape[1]) if targets[0, i] == 1 and targets[1, i] == -1]
    class4 = [inputs[:, i] for i in range(inputs.shape[1]) if targets[0, i] == 1 and targets[1, i] == 1]

    fig, ax = plt.subplots()

    def anim(i):
        
        quiver_1 = ax.quiver(0, 0, 0, 0)
        quiver_2 = ax.quiver(0, 0, 0, 0)
        
        ax.clear()
        
        plt.scatter(np.vstack(class1).T[0,:], np.vstack(class1).T[1,:], marker='s', label = 'Class 1')
        plt.scatter(np.vstack(class2).T[0,:], np.vstack(class2).T[1,:], marker='o', label = 'Class 2')
        plt.scatter(np.vstack(class3).T[0,:], np.vstack(class3).T[1,:], marker='x', label = 'Class 3')
        plt.scatter(np.vstack(class4).T[0,:], np.vstack(class4).T[1,:], marker='P', label = 'Class 4')
                
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        
        w_h1, w_h2 = param[i][0][0,:], param[i][0][1,:]
        b_h1, b_h2 = param[i][1][0].item(), param[i][1][1].item()
        
        db1_slope = (-w_h1[0]) / (w_h1[1])
        db2_slope = (-w_h2[0]) / (w_h2[1])
        
        db1_int = (-b_h1 / w_h1[1])
        db2_int = (-b_h2 / w_h2[1])
        
        quiver_1.set_UVC(np.array([w_h1[0]]), np.array([w_h1[1]]))
        ax.add_artist(quiver_1)
        ax.axline((0, db1_int), slope=db1_slope)
        
        quiver_2.set_UVC(np.array([w_h2[0]]), np.array([w_h2[1]]))
        ax.add_artist(quiver_2)
        ax.axline((0, db2_int), slope=db2_slope)
        
        plt.ylabel('$P2$')
        plt.xlabel('$P1$')
        plt.title('ADALINE')
        plt.legend(loc = 'upper right')
        
        return quiver_1, quiver_2,

    anim = FuncAnimation(fig, anim, frames=len(inputs))
       
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    anim.save(os.path.join('plots', 'adaline_anim.gif'), writer='ffmpeg', fps =10)



# fig, ax = plt.subplots()

# plt.scatter(p[0, :2], p[1, :2], marker='s')
# plt.scatter(p[0, 2:4], p[1, 2:4], marker='o')
# plt.scatter(p[0, 4:6], p[1, 4:6], marker='x')
# plt.scatter(p[0, 6:8], p[1, 6:8], marker='P')

# ax.set_xlim([-4, 4])
# ax.set_ylim([-4, 4])

# quiver_1 = ax.quiver(0, 0, 0, 0)
# quiver_2 = ax.quiver(0, 0, 0, 0)

# def animate(i):
    
#     ax.clear()
    
#     ax.set_xlim([-4, 4])
#     ax.set_ylim([-4, 4])
    
#     plt.scatter(p[0, :2], p[1, :2], marker='s', label = 'Class 1')
#     plt.scatter(p[0, 2:4], p[1, 2:4], marker='o', label = 'Class 2')
#     plt.scatter(p[0, 4:6], p[1, 4:6], marker='x', label = 'Class 3')
#     plt.scatter(p[0, 6:8], p[1, 6:8], marker='P', label = 'Class 4')
    
#     w_h1, w_h2 = w_h[i][0], w_h[i][1]
#     b_h1, b_h2 = b_h[i][0], b_h[i][1]
    
#     db1_slope = (-w_h1[0]) / (w_h1[1])
#     db2_slope = (-w_h2[0]) / (w_h2[1])
    
#     db1_int = (-b_h1[0] / w_h1[1])
#     db2_int = (-b_h2[0] / w_h2[1])
    
#     quiver_1.set_UVC(np.array([w_h1[0]]), np.array([w_h1[1]]))
#     ax.add_artist(quiver_1)
#     ax.axline((0, db1_int), slope=db1_slope)
    
#     quiver_2.set_UVC(np.array([w_h2[0]]), np.array([w_h2[1]]))
#     ax.add_artist(quiver_2)
#     ax.axline((0, db2_int), slope=db2_slope)
    
#     plt.ylabel('$Y$')
#     plt.xlabel('$X$')
#     plt.title('ADALINE')
#     plt.legend(loc = 'upper right')
    
#     return quiver_1, quiver_2,

# anim = FuncAnimation(fig, animate, frames=160)

# anim.save('ADALINE_classification.gif', writer='ffmpeg', fps =10)


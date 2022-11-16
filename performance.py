import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

style.use('dark_background')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def smooth(y):
    points = 5
    box = np.ones(points)/points
    smooth = np.convolve(y, box, mode='same')
    return smooth

def animate(foo):
    try:
        data = [[float(i) for i in x.split()] for x in open('money.txt','r').read().split('\n')]
        niftyp, bankp, nifty, bank = data
        
        ax.clear()
        ax.plot(nifty,  label='Nifty True')
        ax.plot(bank,   label='Bank True')
        ax.plot(niftyp, label='Nifty Pred')
        ax.plot(bankp,  label='Bank Pred')
        
        plt.legend()

    except Exception:
        pass

ani = animation.FuncAnimation(fig, animate, interval=5000)
plt.show()

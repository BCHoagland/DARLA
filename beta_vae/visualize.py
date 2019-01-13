import numpy as np
from visdom import Visdom

viz = Visdom()

title = 'ß-VAE Loss by Epoch'
win = None

def update_viz(epoch, loss):
    global win, title

    if win is None:
        title = title

        win = viz.line(
            X=np.array([epoch]),
            Y=np.array([loss]),
            win=title,
            opts=dict(
                title=title,
                fillarea=True
            )
        )
    else:
        viz.line(
            X=np.array([epoch]),
            Y=np.array([loss]),
            win=win,
            update='append'
        )

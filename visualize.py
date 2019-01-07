import numpy as np
from visdom import Visdom

viz = Visdom()

title = 'Autoencoder Loss'
win = None

def update_viz(epoch, loss, name):
    global win, title

    if win is None:
        title = title

        win = viz.line(
            X=np.array([epoch]),
            Y=np.array([loss]),
            win=title,
            name=name,
            opts=dict(
                title=title
            )
        )
    else:
        viz.line(
            X=np.array([epoch]),
            Y=np.array([loss]),
            win=win,
            name=name,
            update='append'
        )

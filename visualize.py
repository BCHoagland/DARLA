from visdom import Visdom

viz = Visdom()

def display_image(img):
    viz.image(
        img=img,
        win='image',
        opts=dict(
            title='preprocessed image'
        )
    )

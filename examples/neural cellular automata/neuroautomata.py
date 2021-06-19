import pygame as pg
from random import random as r
from neurobiba import Weights, load_weights, save_weights
import copy
import itertools

W, H, size = 100, 60, 10

pg.init()
screen = pg.display.set_mode((W*size, H*size), 0, 32)
pg.display.set_caption('CYBERBIBA')

def update():
    nn = Weights([27,3])
    canvas1 = [[[r(),r(),r()] for y in range(H)] for x in range(W)]
    canvas2 = copy.deepcopy(canvas1)
    return nn, canvas1, canvas2

nn, canvas1, canvas2 = update()
is_running = True
while is_running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            is_running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_r:
                nn, canvas1, canvas2 = update()
            if event.key == pg.K_s:
                save_weights(nn, "weights")
            if event.key == pg.K_l:
                nn, canvas1, canvas2 = update()
                nn = load_weights("weights")

    for x, i in enumerate(canvas1):
        for y, _ in enumerate(i):
            neighbors = [canvas1[(x+dx-1)%W][(y+dy-1)%H] for dy in range(3) for dx in range(3)]
            neighbors = list(itertools.chain(*neighbors))
            result = nn.feed_forward(neighbors)
            canvas2[x][y] = result
            color = tuple(map(lambda x: int(x*255),result))
            screen.fill(color, (x*size, y*size, size, size))

    canvas1, canvas2 = canvas2, canvas1
    pg.display.flip()
    
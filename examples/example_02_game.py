import pygame as pg
import sys

from neurobiba import *

WIDTH, HEIGHT = 40, 40

PX = 10

SCREEN_SIZE = SCR_WIDTH, SCR_HEIGHT = WIDTH * PX, HEIGHT * PX

pg.init()
clock = pg.time.Clock()
screen = pg.display.set_mode(SCREEN_SIZE)

weights = create_weights([2, 1])

# list of { "point": [float, float], "value": bool }
dataset = []

active = False

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        if event.type == pg.MOUSEBUTTONUP:
            x, y = pg.mouse.get_pos()
            x = x / WIDTH / PX
            y = y / HEIGHT / PX
            dataset.append({"point": (x, y), "value": active})
            print(dataset[-1])
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_n:
                weights = create_weights([2, 10, 10, 10, 1])
            if event.key == pg.K_x:
                active = not active
            if event.key == pg.K_DELETE:
                dataset = []

    for _ in range(50):
        for data in dataset:
            point = data["point"]
            weights = training(point, data["value"], weights)

    for x in range(WIDTH):
        for y in range(HEIGHT):
            result = feed_forward([x / WIDTH, y / HEIGHT], weights)
            color = tuple([result * 255] * 3)
            screen.fill(color, (x * PX, y * PX, PX, PX))

    for data in dataset:
        color = (255, 255, 255) if data["value"] else (0, 0, 0)
        x = data["point"][0] * WIDTH * PX
        y = data["point"][1] * HEIGHT * PX
        pg.draw.circle(screen, (255, 0, 0), (x, y), 7)
        pg.draw.circle(screen, color, (x, y), 5)

    color = (255, 255, 255) if active else (0, 0, 0)
    screen.fill(color, (5, 5, 20, 20))

    pg.display.flip()
    clock.tick(60)

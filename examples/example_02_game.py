import pygame as pg
import sys

from neurobiba import *

WIDTH, HEIGHT = 40, 40

PX = 10

SCREEN_SIZE = SCR_WIDTH, SCR_HEIGHT = WIDTH * PX, HEIGHT * PX

pg.init()
screen = pg.display.set_mode(SCREEN_SIZE)

NN_SIZE = [2, 10, 10, 1]
weights = create_weights(NN_SIZE)

# list of { "point": [float, float], "value": bool }
dataset = []

active = False

basicFont = pg.font.SysFont('arial', 15, bold=True)
def write_on_screen (text, x, y):
    text = basicFont.render(text, True, (255, 0, 0))
    textRect = text.get_rect()
    textRect.left = x
    textRect.centery = y
    screen.blit(text, textRect)


while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
        if event.type == pg.MOUSEBUTTONUP:
            x, y = pg.mouse.get_pos()
            x = x / SCR_WIDTH
            y = y / SCR_HEIGHT
            dataset.append({"point": (x, y), "value": active})
            print(dataset[-1])
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_x:
                active = not active
            if event.key == pg.K_n:
                weights = create_weights(NN_SIZE)
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

    screen.fill((255, 0, 0), (4, 4, 22, 22))
    color = (255, 255, 255) if active else (0, 0, 0)
    screen.fill(color, (5, 5, 20, 20))

    write_on_screen('MOUSE : set point', 30, 10)
    write_on_screen('X : switch color', 30, 25)
    write_on_screen('DEL : clear dataset', 30, 40)
    write_on_screen('N : reset weights', 30, 55)

    for data in dataset:
        color = (255, 255, 255) if data["value"] else (0, 0, 0)
        x = int(data["point"][0] * WIDTH * PX)
        y = int(data["point"][1] * HEIGHT * PX)
        pg.draw.circle(screen, (255, 0, 0), (x, y), 7)
        pg.draw.circle(screen, color, (x, y), 5)

    pg.display.flip()


import pygame as pg
import sys

from neurobiba import *

WIDTH, HEIGHT = 40, 40

PIXEL = 10

ACCENT_COLOR = 200, 0, 0

SCREEN_SIZE = SCR_WIDTH, SCR_HEIGHT = WIDTH * PIXEL, HEIGHT * PIXEL

pg.init()
screen = pg.display.set_mode(SCREEN_SIZE)

NN_SIZE = [2, 10, 10, 1]
weights = create_weights(NN_SIZE)

# list of { "point": [float, float], "value": bool }
dataset = []

active = False

basicFont = pg.font.SysFont(pg.font.get_fonts()[0], 15, bold=True)


def draw_text(surface, x, y, font, text):
    for i, string in enumerate(text):
        text = font.render(string, True, ACCENT_COLOR)
        surface.blit(text, (x, y + i * 15))


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
            screen.fill(color, (x * PIXEL, y * PIXEL, PIXEL, PIXEL))

    screen.fill(ACCENT_COLOR, (5, 5, 24, 24))
    color = (255, 255, 255) if active else (0, 0, 0)
    screen.fill(color, (7, 7, 20, 20))

    draw_text(screen, 35, 2, basicFont, [
        'MOUSE : set point',
        'X : switch color',
        'DEL : clear dataset',
        'N : reset weights'
    ])

    for data in dataset:
        color = (255, 255, 255) if data["value"] else (0, 0, 0)
        x = int(data["point"][0] * WIDTH * PIXEL)
        y = int(data["point"][1] * HEIGHT * PIXEL)
        pg.draw.circle(screen, ACCENT_COLOR, (x, y), 7)
        pg.draw.circle(screen, color, (x, y), 5)

    pg.display.flip()

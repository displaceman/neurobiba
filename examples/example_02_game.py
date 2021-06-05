import neurobiba as nb

import pygame as pg
import pygame_gui as pgui


WIDTH, HEIGHT = 40, 40
PIXEL = 10
SIDEBAR_WIDTH = 200
ACCENT_COLOR = 200, 0, 0
WEIGHTS_SHAPE = [2, 10, 10, 1]

IMAGE_SIZE = IMG_WIDTH, IMG_HEIGHT = WIDTH * PIXEL, HEIGHT * PIXEL
SCREEN_SIZE = SCR_WIDTH, SCR_HEIGHT = IMG_WIDTH + SIDEBAR_WIDTH, IMG_HEIGHT

pg.init()
pg.display.set_caption('neurobiba')
window_surface = pg.display.set_mode(SCREEN_SIZE)

manager = pgui.UIManager((800, 600))

reset_weights_btn = pgui.elements.UIButton(relative_rect=pg.Rect((5, 5), (190, 30)),
                                           text='reset weights',
                                           manager=manager)

clear_data_btn = pgui.elements.UIButton(relative_rect=pg.Rect((5, 40), (190, 30)),
                                        text='delete all points',
                                        manager=manager)

pop_data_btn = pgui.elements.UIButton(relative_rect=pg.Rect((5, 75), (190, 30)),
                                      text='delete prev point',
                                      manager=manager)

switch_color_btn = pgui.elements.UIButton(relative_rect=pg.Rect((5, 110), (190, 30)),
                                          text='switch color',
                                          manager=manager)

weights = nb.Weights(WEIGHTS_SHAPE)

# list of { "point": [float, float], "value": bool }
dataset = []

black = False

basicFont = pg.font.SysFont(pg.font.get_fonts()[0], 15, bold=True)


def draw_text(surface, x, y, font, text):
    for i, string in enumerate(text):
        text = font.render(string, True, ACCENT_COLOR)
        surface.blit(text, (x, y + i * 15))


clock = pg.time.Clock()

is_running = True

while is_running:
    time_delta = clock.tick(60) / 1000.0
    for event in pg.event.get():
        if event.type == pg.QUIT:
            is_running = False

        if event.type == pg.USEREVENT:
            if event.user_type == pgui.UI_BUTTON_PRESSED:
                if event.ui_element == reset_weights_btn:
                    weights = nb.Weights(WEIGHTS_SHAPE)
                if event.ui_element == clear_data_btn:
                    dataset = []
                if event.ui_element == pop_data_btn:
                    if len(dataset):
                        dataset.pop()
                if event.ui_element == switch_color_btn:
                    black = not black

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_x:
                black = not black

        if event.type == pg.MOUSEBUTTONUP:
            mouse = x, y = pg.mouse.get_pos()
            if pg.Rect((SIDEBAR_WIDTH, 0), (SCR_WIDTH, SCR_HEIGHT)).collidepoint(mouse):
                x = (x - SIDEBAR_WIDTH) / IMG_WIDTH
                y = y / IMG_HEIGHT
                dataset.append({"point": (x, y), "value": black})
                print(dataset[-1])

        manager.process_events(event)

    manager.update(time_delta)

    for _ in range(50):
        for data in dataset:
            point = data["point"]
            weights.training(point, data["value"])

    image = pg.Surface((WIDTH * PIXEL, HEIGHT * PIXEL))

    for x in range(WIDTH):
        for y in range(HEIGHT):
            result = weights.feed_forward([x / WIDTH, y / HEIGHT])
            color = tuple([result * 255] * 3)
            rect = (x * PIXEL, y * PIXEL, PIXEL, PIXEL)
            # window_surface.fill(color, rect)
            image.fill(color, rect)

    image.fill(ACCENT_COLOR, (5, 5, 24, 24))
    color = (255, 255, 255) if black else (0, 0, 0)
    image.fill(color, (7, 7, 20, 20))

    for data in dataset:
        color = (255, 255, 255) if data["value"] else (0, 0, 0)
        x = int(data["point"][0] * WIDTH * PIXEL)
        y = int(data["point"][1] * HEIGHT * PIXEL)
        pg.draw.circle(image, ACCENT_COLOR, (x, y), 7)
        pg.draw.circle(image, color, (x, y), 5)

    window_surface.blit(image, (SIDEBAR_WIDTH, 0))

    manager.draw_ui(window_surface)

    pg.display.flip()

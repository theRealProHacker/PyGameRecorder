from pygame_screen_record.EventRegister import EventRegister, pg, time, random

pg.init()

screen = pg.display.set_mode((900, 600))
font = pg.freetype.SysFont(None, 70)

reg = EventRegister("out", "events_ex2.json").seed()

text = "0"
current_center = screen.get_rect().center

try:
    running = True
    while running:
        time.sleep(0.0099)  # 100 fps
        for event in (events := reg.get_events()):
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                text = str(random.randint(1, 100))
            elif event.type == pg.MOUSEMOTION:
                current_center = event.pos
        if running:
            screen.fill((255,) * 3)
            frect = font.get_rect(text)
            frect.center = current_center
            font.render_to(screen, frect, None, fgcolor=(10, 120, 200))
            pg.display.flip()

finally:
    pg.quit()
    reg.save()

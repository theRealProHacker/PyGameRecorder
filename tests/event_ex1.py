import sys,os
os.chdir(os.path.dirname(__file__))
sys.path.append("../")

from EventRegister import EventRegister, pg, time

pg.init()

screen = pg.display.set_mode((900,600))
font = pg.freetype.SysFont(None,70)

reg = EventRegister("out","events_ex1.json")

text = ""

try:
    running = True
    while running:
        time.sleep(0.0099) # 100 fps
        for event in (events := reg.get_events()):
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
                    running = False
                elif event.key == pg.K_BACKSPACE:
                    text = text[:-1]
                else:
                    text+=event.unicode
                    print(event.unicode, event.key, event.mod, event.key - event.mod, pg.key.get_pressed()[event.key])
        if running:
            screen.fill((255,)*3)
            frect = font.get_rect(text)
            frect.center = screen.get_rect().center
            font.render_to(screen, frect, None, fgcolor=(10,120,200))
            pg.display.flip()

finally:
    pg.quit()
    reg.save()
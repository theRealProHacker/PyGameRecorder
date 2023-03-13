""" 
This file is for showcasing the ScreenRecorder. It uses the code from the pygame examples with slight changes:

pygame.examples.liquid
This example demonstrates a simplish water effect of an
image. It attempts to create a hardware display surface that
can use pageflipping for faster updates. Note that the colormap
from the loaded GIF image is copied to the colormap for the
display surface.
This is based on the demo named F2KWarp by Brad Graham of Freedom2000
done in BlitzBasic. I was just translating the BlitzBasic code to
pygame to compare the results. I didn't bother porting the text and
sound stuff, that's an easy enough challenge for the reader :]
"""

from sys import getsizeof
import pygame as pg
from math import sin
import time
from pygame_screen_record import ScreenRecorder, RecordingPlayer, add_codec
import logging

add_codec("mp4", "mpv4")

logging.basicConfig(level=logging.DEBUG)

running = True


def main():
    global running
    # initialize and setup screen
    pg.init()
    screen = pg.display.set_mode((640, 480))

    # load image and quadruple
    imagename = "liquid.bmp"
    bitmap = pg.image.load(imagename)
    bitmap = pg.transform.scale2x(bitmap)
    bitmap = pg.transform.scale2x(bitmap)

    # get the image and screen in the same format
    if screen.get_bitsize() == 8:
        screen.set_palette(bitmap.get_palette())
    else:
        bitmap = bitmap.convert()

    # prep some variables
    anim = 0.0

    # mainloop
    xblocks = range(0, 640, 20)
    yblocks = range(0, 480, 20)

    """ 
    Here we start the recording. 
    Play around with the fps and the compress to get an impression on the change that makes
    """
    recorder = ScreenRecorder(24, compress=1)
    recorder.start_rec()

    try:
        running = True
        while running:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running = False
                elif e.type == pg.KEYDOWN:
                    running = False

            anim = anim + 0.02
            for x in xblocks:
                xpos = (x + (sin(anim + x * 0.01) * 15)) + 20
                for y in yblocks:
                    ypos = (y + (sin(anim + y * 0.01) * 15)) + 20
                    screen.blit(bitmap, (x, y), (xpos, ypos, 20, 20))
            if not running:
                screen.fill((0,) * 3)
            pg.display.flip()
            time.sleep(0.01)
    finally:
        recorder.stop_rec()

    """ 
    Now the recording has finished and we save the recording. 
    The screen will be black as long as the saving takes
    """
    recording = recorder.get_single_recording()
    print(getsizeof(recording))
    recording.save(("my_recording1", "mp4"))

    player = RecordingPlayer(recording)
    player.play()
    try:
        running = True
        while running:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    running = False
            pg.display.flip()
            time.sleep(0.01)
    finally:
        player.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        pg.quit()

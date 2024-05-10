# A Simple pygame ScreenRecorder

## Why you should use pygame ScreenRecorder?

1. Relatively high accuracy
2. No (noticable) performance issues
3. Recording FPS is not bound to game FPS
4. Straightforward usage
5. The codebase is well commented and typed

## Dependencies

Apart from pygame and python >=3.8

1. opencv-python (includes numpy)
2. FFmpeg if you want to save videos

## Install

`pip install pygame-screen-record`

## FAQ

See the FAQ [in the wiki](https://github.com/theRealProHacker/PyGameRecorder/wiki/FAQ)

## To-Dos

2. Event recording (mouse, key, quit, etc.) ✔️
3. Sound recording (Either event based with function hooking or as numpy arrays)
4. A proper video player
5. Add a wiki ✔️

## Contributing

File any bugs or feature requests as GitHub issues. Any comments are always welcome. 

# How To Use

Probably you just want to make a recording of their game and save it in a video file. Here comes how.

> Note: I am using `pg` as an alias for `pygame` in the following code snippets

A typical game script might look like this:

```py
import pygame as pg
pg.init()

init_code()

try:
    while True:
        event_handling()
        updating()
        drawing()
finally:
    clean_up()
    pg.quit()
```

The `try - finally - statement` is very important to catch any exceptions and clean up whether the game ended naturely or not.

Adding a recorder is very simple:

```py
import pygame as pg
from pygame_screen_record import ScreenRecorder

pg.init()

init_code()

recorder = ScreenRecorder(60) # pass your desired fps
recorder.start_rec() # start recording

try:
    while True:
        event_handling()
        updating()
        drawing()
finally:
    recorder.stop_rec()	# stop recording
    recorder.save_recording("my_recording.mp4") # saves the last recording
    clean_up()
    pg.quit()
```

This code will record your screen the whole game and then save it in the current working directory as `my_recording.mp4`.
Typical values for the frames per second (fps) are 24 (for slow games), 30, 60 and 120 (Most displays only refresh at 60 Hertz, so most users won't see a difference upwards of 60 fps). Don't forget that the fps value is (at least in theory) proportional to the memory consumption of your recording.

One cool thing of many is that you can chain functions. So for example

```py
recorder = ScreenRecorder(60)
recorder.start_rec()
```
is equivalent to

```py
recorder = ScreenRecorder(60).start_rec()
```

and you can also just write: `recorder.stop_rec().save_recording("my_recording.mp4")`

# Advanced Recording Options

Some of the options available:

1. Record multiple recordings
2. Record any surface
3. Compress `ScreenRecorders`
4. Stream recordings (maybe add frame streaming too?)
5. Apply effects on recordings

## Record multiple recordings

In the example, we only recorded a single recording. However, you can record as many recordings as you like with a single `ScreenRecorder`. To stop and start recordings use `stop_rec` and `start_rec`. You can also abort recordings using `abort_rec`. 

Finally, when you want to save your recordings call `save_recordings("mp4")`. More detailed saving options are given in the next chapter [Advanced Saving Options](#advanced-saving-options). 

## Record any `Surface`

In most cases you want to record the whole screen. But, you can also pass an optional argument `surf` to a `ScreenRecorder`.

```python
my_surface = pg.surface.Surface((900, 600))
recorder = ScreenRecorder(60, my_surface)
```

## Compress recordings

You can choose to compress your recordings like this

```py
recorder = ScreenRecorder(compress=2) # fps defaults to 60 
```

What does that mean? It means that every frame will be scaled down `2`-times by two. This reduces the total memory consumption by `2^(2^2) = 16`! But normally you will only compress by one or not compress at all. The recordings will automatically be decompressed when played or saved. So don't worry about that. Just try out whether the loss in resolution is okay for your needs.

## Stream recordings

A stream in this sense is any object that implements a `send` function that can take a recording. To set a stream pass it to the `ScreenRecorder` constructor.

```python
class Stream:
    def send(self, rec):
        print(f"Recording received with {rec.frame_number} frames, a size of {rec.size} and a total length of {rec.length} s")

my_stream = Stream()
recorder = ScreenRecorder(stream=my_stream)
```

Now `recorder.stop_rec()` will send to that stream and also save the recording internally. With `recorder.stop_rec_to_stream(stream = None)` you send to the stream without saving and can optionally specify a stream that will override the recorders default stream.

## Set individual recordings fps

```py
ScreenRecorder(60).start_rec(30)
```

will record at 30 fps for this one recording.

## Get all recordings of a recorder

```py
all_recordings = recorder.get_recordings()
```

## Attributes of a recorder

These are the attributes of a `ScreenRecorder` instance. Don't change any of these if you don't have a reason! Create a new recorder instead.

`fps: float`  
selfexp.

`surf: pg.Surface`  
selfexp.

`compress: int`  
selfexp.

`size: Tuple[int,int]`  
The size (width, height) of the recorded surface. Change this attribute only if you are also manually changing the surface at the same time.

`dt: float`  
The time between (delta time) two frames in ms.  
`dt = 1000/fps`

`recordings: List[Recording]`  
selfexp. Same as `get_recordings()`

## Post-Processing a recording

1. Add frames
2. Resize
3. Apply effects

## Adding frames

You can always append frames to a recording:

`recording.add_frame(frame: pg.Surface)`

## Resize a recording

You might need to rescale a whole recording to a specific size:

`recording.resize(size: Tuple[int,int])`

## Apply effects

If there is more to do than just resizing:

`recording.apply(effect: Callable[[pg.Surface], pg.Surface])`

Will apply the effect on every frame of the recording.

## Attributes of a recording

These are pretty much the same as the attributes of a ScreenRecorder

`fps: float`  
selfexp.

`surf: pg.Surface`  
selfexp.

`compress: int`  
selfexp.

`size: Tuple[int, int]`
selfexp. Change this attribute only if you are also manually changing the frames at the same time (e.g. Applying a resizing filter).

`dt: float`  
selfexp.

`frames: List[pg.Surface]`  
selfexp.

# Advanced Saving Options

Introducing the `RecordingSaver`

`RecordingSaver(recordings: List[Recording], key: str | Sequence[str] | Callable[[int], Optional[str]], save_dir: AnyPath = None, blocking: bool = True)`

```py
recordings = recorder.get_recordings()
saver = RecordingSaver(recordings, "mp4", "saved_files")
saver.save()
```

Saves the given recordings as `mp4` files in `./saved_files`. But you can also just call

```py
saved_recordings = recorder.save_recordings("mp4", "saved_files")
```

## Explanation

- `key`
    You can either give a str, a list or a function  
    If key is a str that determines the format of the recordings and they will be saved as `recording_0.{key}`,`recording_1.{key}`, etc. 
    
    Valid formats are listed if you call `available_formats()`. You can add/update FFmpeg-supported formats by calling `add_codec(format:str, codec: int | str)`.  

    Maybe its worth to mention the `npz` file format. It is not a classical video format but actually a way to save numpy arrays (npz = **n**um**p**y **z**ipped). If you don't need to share the recording in the internet or so, this is an efficient alternative. Also this library has built-in support for replaying these files.    

    If key is a sequence then the ith recording will be saved as the ith element of the recording. If an element is `None` the according recording will be skipped.   

    It is a very similar case if you give a function. The function gets an int passed and should return `None` or a filename.

    An example for such a key is

    ```python
    key = lambda x: if x%2 == 0 then None else ("recording_{x}","mp4")
    ```

    This will return `None` (and skip the save) for every recording with an even index. 
- `save_dir`  
The directory where the recordings will be saved. Defaults to the current working directory
- `blocking`  
Whether the save should block. Defaults to `True`

The save returns a list of paths to the recordings in the given directory. This list will not **always** be the same length as the recordings in the `Recorder` but will only return a list of recordings that were actually saved. 

However, if you set `block = False` the function will return another function that returns the list of paths and must be called before the script ends! Now you might ask yourself why that makes any sense. Here is an example

```python
# At this point we have a recorder that recorded some recordings
# Now we want to save the recordings as `mp4` and also as `npz`
import time #to measure how long the saves took

# A naive approach is this
start = time.time()
recorder.save_recordings("mp4", "saved_files1")
recorder.save_recordings("npz", "saved_files1")
print("First save took:", time.time() - start)

# Now we use non-blocking (asynchronous) code
start = time.time()
join1 = recorder.save_recordings("mp4", "saved_files2", False)
join2 = recorder.save_recordings("npz", "saved_files2", False)
print("This message doen't have to wait for the save. Instead it comes almost instantly")

time.sleep(2) # We add some more virtual io with time.sleep

# Finally we join the save
join1()
join2()
print("Second save took:", time.time() - start)
```

The second option is favorable because it takes less time than the first. 

## Memory Cosniderations

We talked about how to efficiently save your recordings (from a time aspect). But now we talk about how you can reduce memory consumption. Generally, all video recordings will be automatically compressed by FFmpeg/numpy. However, there are three ways you can reduce memory consumption:  
1. Reduce fps. One cool thing about this ScreenRecorder is that you can record at a different framerate than you play the game. For example you can have a game frame rate of 60 fps but only record at 30 fps. This would halve the memory usage in comparison to if you recorded at 60 fps. 
1. Resize the recording. We already established that you can halve the recording size as often as you like. But this will only reduce memory usage while the program runs. The result will still be saved in the original size. However, you can save the recording in a smaller size by using  
    `recording.resize(pg.Vector2(recording.size)/your_scale_factor)`  
    This will actually save and play the recording in that size, which might look very weird. 
    So you might not actually want to do that. 
1. Shorten the recording length. You can cut out parts of a recording like this. Lets say you only want the first 300 frames.

    ```python
    # This will actually mutate the recording. 
    recording[0:300]
    ``` 
1. Lastly, you can reduce the depth of the recording by reducing the depth of the recorded screen `pg.display.set_mode((900,600), depth=your_depth)`. This will definitely reduce the memory usage while the program is running and it might also reduce the memory usage on the disk. However, decreasing the depth of the screen will also decrease the variety in color. But in most amateur applications this might just not matter anyway because you are not using very nuanced colors.   

# Replay recordings

## First note
I had already implemented a VideoPlayer that could play a Full HD video (1920x816) pretty well (Thats over 6 MB per frame at 24 fps). However, there were several issues (without much detail):
1. Missing sound
1. Memory
1. Lags/Preloading (combined with Memory)

Finally, I came to the conclusion that it makes no sense to write a VideoPlayer in pure Python. 

## Easiest way to replay a recording

```py
# easiest
player = RecordingPlayer(recording).play()

# with an on_stop callback
def on_stop(): 
    print("Playing finished")
player = RecordingPlayer(recording, on_stop).play()

# with a different surface
my_surface = pg.Surface((900,600))
player = RecordingPlayer(recording, None, my_surface).play()
```

Make sure that you are not blitting anything else to the surface. However, you still have to do the flipping/updating yourself (I figured it would be weird if the player did that for you). 

Very important is that you always `stop` a player in your `finally-clause` even if you normally wait for the player to end. Here a contextmanager might make sense too but I'm tending to rather no. 

# Advanced `RecordingPlayer` Options

## Pausing
You already know how to start a player. You can also pause the player with `pause`. Playing is also unpausing.

```python
player.pause() 
```

## Stopping
This stops the player. As said above, always stop the player if in doubt. However, don't even try to reuse a manually stopped player. Just make a new player instead, it's really simple.

```python
player.stop()
```

## Seeking
Seeking is known from files and means going to a certain position. 

```python
player.seek(300) # goes to frame 300 / the 301th frame
player.seekms(3000.0) # goes to second 3 of the recording
```

## Telling
Similarly telling is also known from files and means getting the current position. 

```python
player.tell() # Gets the current position
player.tellms() # Gets the current position in milliseconds
```

## Restarting
This method is a mixture between reviving the player after it stopped and just seeking the very first frame and playing.

```python
player.restart()
```

## Getting state information
The player has a `is_` function. There are two reasons for the name
1. It resolves the conflict with the python keyword `is`
1. It might make the code more readable, reading `player.is_("playing")` is easy to understand and nicer to implement than making individual function for every possible state

```python
player.is_("started") 
player.is_("stopped")  
player.is_("playing")
player.is_("paused")
```

They are all pretty self explanatory. But remember two things:
1. `player.is_("stopped")` might be the most important state because you shouldn't call any other function when the player is stopped (Except restart and stop).
1. `is_("paused")` is **not** equal to `not is_("playing")` (In the beginning the player is neither playing nor paused)

## Making use of the `on_stop`

The `on_stop` is a very powerful tool because callbacks are always cool. Optionally, `on_stop` will be passed the player object itself. So, you can really do anything you want. I wrote three example callbacks which are very likely to be useful to the API user. Don't forget to import them before you use them (They are not included in `*`). 

1. `play_indefinite` will restart the player indefinitely as long as the player is not stopped manually (Stopping the player manually will overwrite the on_stop).

    ```python
    player = RecordingPlayer(recording,play_indefinite).play()
    ``` 
1. `play_n_times` plays the player `n` times.

    ```python
    player = RecordingPlayer(recording,play_n_times(5)).play() # plays 5 times
    ```
1. `play_n_wrapper` plays the player `n` times but it wraps another function to call each time. To come back to our very first `on_stop`: 

    ```python
    @play_n_wrapper(5)
    def on_stop(): 
        print("Playing finished")
    player = RecordingPlayer(recording,on_stop).play()
    ```

## Playing saved npz files

I showed you how to save your recordings as `npz` files. However, you also need to know how to play them back. For this there is a `NPZPlayer` class. It takes a path to a file and extra parameters just like the `RecordingPlayer`

```python
player = NPZPlayer("my_npz_file.npz", on_stop=my_on_stop).play()
```

In all other regards it is the same as the `RecprdingPlayer`

# Event Register
One of my to-dos was an event register. This task is accomplished. Here comes the tutorial for this. 

Let's suppose you are using events and have a deterministic game (No randomness/randomness with a seed). You just need to do four things to record your game. 
1. `import EventRegister from EventRegister`
1. Create a new `EventRegister` object
1. Get your events from the object
1. Finally, save the registered events.

## Example

```python
import EventRegister from EventRegister

pg.init()

reg = EventRegister("in","events.json") # save as json

try:
    running = True
    while running:
        time.sleep(0.0099) # 100 fps
        for event in reg.get_events(): # instead of pg.event.get()
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                print(event.button,event.pos)
finally:
    pg.quit()
    reg.save()
```

Now to replay that exact recorded game. Just swap `in` with `out` when instanciating the `reg` object and everything should work exactly as expected.

## Random Seeds

If your game uses randomness - which most games should - it's very simple. 
This will automatically load or save the seed depending on the mode.  

```python
reg = EventRegister("in","events.json").seed()
```
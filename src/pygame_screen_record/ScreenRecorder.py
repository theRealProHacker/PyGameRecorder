import atexit
import logging
import os
import time
import threading
from contextlib import contextmanager
from functools import wraps
from threading import Thread
from typing import Callable, Coroutine, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pygame as pg

##############################################################################################################
# Helpers


def mt(*_, **__):
    pass


# types


def ensure_type(var, _type):
    if not isinstance(var, _type):
        raise TypeError(f"{var=} must by of type {_type=}")


AnyPath = Union[str, bytes, os.PathLike]

# Test utils


def timer_wrapper(func, info: str = ""):
    @wraps(func)
    def inner(*args, **kwargs):
        start = time.time()
        ret_val = func(*args, **kwargs)
        end = time.time()
        logging.debug("Finished:", func.__name__, end - start, info)
        return ret_val

    return inner


# time helper functions; all in milliseconds


def _get_time():
    return time.time() * 1000


def _time_since(t): return _get_time() - t
def _sleep(t): return time.sleep(t / 1000)

# Video Converting:


def _pg_to_cv2(arr): return cv2.cvtColor(arr.swapaxes(0, 1), cv2.COLOR_RGB2BGR)


def _surf_to_arr(frame: pg.surface.Surface):
    """Internal function to convert a pg.surface.Surface to a np.ndarray depending on the depth.
    See the pygame documentation for information on the surfarray submodule"""
    try:
        pg_frame = pg.surfarray.pixels3d(
            frame
        )  # convert the surface to a np array. Only works with depth 24 or 32, not less
    except:
        pg_frame = pg.surfarray.array3d(
            frame
        )  # convert the surface to a np array. Works with any depth
    return pg_frame


def cleanup():
    for k, v in globals().items():
        if isinstance(v, RecordingPlayer):
            if v.is_("playing"):
                logging.warning(f"Stopping not ended player ({k}) ...")
                v.stop()
        elif isinstance(v, ScreenRecorder):
            if v.running_thread is not None:
                logging.warning(f"Stopping not ended recording ({k}) ...")
                v.running_thread.stop()
    cv2.destroyAllWindows()
    for thread in threading.enumerate():
        if not thread.name == "MainThread":
            logging.error(
                "Joining unfinished Thread: "
                + str((thread.name, thread.__class__.__name__))
            )
            thread.join()


##############################################################################################################


##############################################################################################################
# Compression


def compress(frame: pg.surface.Surface, comp_rate: int) -> pg.surface.Surface:
    if comp_rate:
        vec = pg.Vector2(frame.get_size())
        return pg.transform.smoothscale(frame, vec / (2**comp_rate))
    else:
        return frame


def decompress(frame: pg.surface.Surface, compress: int):
    for _ in range(compress):
        frame = pg.transform.scale2x(frame)
    return frame


##############################################################################################################


##############################################################################################################
# Threading


class StoppableThread(Thread):
    """A general StoppableThread. It can be stopped with thread.stop()"""

    def __init__(self, coro_make: Callable[[], Coroutine]):
        """coro_make is a function that returns a Coroutine"""
        Thread.__init__(self)
        self.coro = coro_make()
        self.running: bool = True

    def run(self):
        logging.debug("Started " + self.__class__.__name__)
        while self.running:
            self.coro.__next__()
        self.coro.close()
        logging.debug("Finished " + self.__class__.__name__)

    def stop(self):
        self.running = False


class RecordingThread(StoppableThread):
    def __init__(
        self,
        surf: pg.surface.Surface,
        dt: float,
        save_frames: Callable[[pg.surface.Surface], None],
    ):
        def coro():
            last = _get_time()
            try:
                while True:
                    yield
                    save_frames(surf.copy())
                    diff = dt - _time_since(last)
                    if diff > 0:
                        _sleep(diff)
                    last = _get_time()
            except (pg.error, GeneratorExit):
                pass

        super().__init__(coro)  # give the coroutine to the StoppableThread init


class PlayingThread(StoppableThread):
    """Intern Thread for playing Recordings"""

    def __init__(
        self,
        surf: pg.surface.Surface,
        dt: float,
        get_frames: Callable[[int], pg.surface.Surface],
        on_stop: Callable[[], None],
    ):
        """The init for a PlayingThread

        Parameters
        ----------
        surf: pg.surface.Surface
            The Surface to draw on
        dt: float
            The milliseconds between each frame
        get_frames: (int)->pg.surface.Surface
            The function to get the frame at a specified position
        on_stop: ()->None
            The callback to call when the Thread stops playing
        """
        Thread.__init__(self)
        self.running = True
        self.paused = False
        self.position = 0
        self.surf = surf
        self.dt = dt
        self.get_frames = get_frames
        self.on_stop = on_stop

    def run(self):
        """
        Runs the PlayingThread
        """
        last = _get_time()
        while self.running:
            if not self.paused:
                try:
                    frame = self.get_frames(self.position)
                    self.surf.blit(frame, (0, 0))
                except (IndexError, pg.error):
                    break
                self.position += 1
                diff = self.dt - _time_since(last)
                if not self.running:
                    break
                if diff > 0:
                    _sleep(diff)
                last = _get_time()
        self.running = False
        self.on_stop()

    def play(self):
        self.paused = False

    def pause(self):
        self.paused = True


##############################################################################################################


##############################################################################################################
# Recording


class Recording:
    def __init__(self, fps: float, size: Tuple[int, int], compress: int):
        self.fps = fps
        self.size = size
        self.compress = compress
        self.dt = 1000 / fps
        self.frames: List[pg.surface.Surface] = []
        self.__getitem__ = self.frames.__getitem__  # type: ignore

    def add_frame(self, frame: pg.surface.Surface):
        """Add a frame to the recording"""
        self.frames.append(compress(frame, self.compress))

    @property
    def frame_number(self):
        """Number of frames in the recording"""
        return len(self.frames)

    @property
    def total_length(self):
        """Length of the recording in seconds"""
        return len(self.frames) * self.dt / 1000

    def resize(self, size: Tuple[int, int]):
        """Resize a Recording"""
        if not size == self.size:
            self.frames = [
                pg.transform.smoothscale(frame, size) for frame in self.frames
            ]
            self.size = size
        return self

    def apply(self, effect: Callable[[pg.surface.Surface], pg.surface.Surface]):
        self.frames = [effect(frame) for frame in self.frames]
        return self

    def __getitem__(self, index):
        try:
            self.frames = [*self.frames[index]]  # slice
        except TypeError:
            # self.frames = [self.frames[index]] # index
            raise TypeError("You can only slice frames")
        return self

    def __sizeof__(self) -> int:
        try:
            byte_size = self.frames[0].get_bytesize()
            w, h = self.size
            comp_fact = (self.compress**2) or 1
            return int(self.frame_number * w * h * byte_size / comp_fact)
        except IndexError:
            return 0

    def save(self, filename: str):
        """Saves a single recording given a recording and a filename. Blocks until the recording is saved.

        Parameters
        ----------
        filename: str
            A filename to save to. Available extensions are in the documentation or by calling `available_formats()`

        Raises
        ------
        RuntimeError
            If a video is being saved and ffmpeg is not installed
        """
        logging.debug("Saving recording to: " +
                      os.path.abspath(filename))
        frames: List[np.ndarray] = [
            _surf_to_arr(decompress(frame, self.compress)) for frame in self.frames
        ]
        _save_single(frames, self.fps, self.size, filename)


class ScreenRecorder:
    def __init__(
        self,
        fps: float = 60,
        surf: Optional[pg.surface.Surface] = None,
        compress: int = 0,
        stream=None,
    ):
        """Inits a ScreenRecorder
        @params:
        fps: float
            Frames per second the screen recorder should record (default=60)
        surf: pg.surface.Surface
            The surface that should be recorded (default=pg.display.get_surface())
        compress: int = 0
            Whether to compress the recording and by how much.
        stream: Implements .send()
            A stream that gets a Recording object when the recording is stopped with recorder.stop_rec_to_stream()
        """
        self.fps = fps
        self.surf = surf or pg.display.get_surface()
        self.compress = compress
        self.stream = stream
        assert (
            self.surf is not None
        ), "You haven't set the display mode yet or you haven't inited pygame!"
        self.size = self.surf.get_size()
        self.dt: float = 1000 / self.fps
        self.running_recording: Optional[Recording] = None
        self.running_thread: Optional[RecordingThread] = None
        self.recordings: List[Recording] = []

    def start_rec(self, fps: Optional[float] = None):
        """Starts a recording
        @params:
        fps: overrides the classes default fps for this recording
        """
        assert (
            self.running_thread is None
        ), "You can't start recording without stopping the current recording."
        self.running_recording = Recording(
            fps or self.fps, self.size, self.compress)
        self.running_thread = RecordingThread(
            self.surf, self.running_recording.dt, self.running_recording.add_frame
        )
        self.running_thread.start()
        return self

    def _stop_rec(self):
        if self.running_thread is not None:
            self.running_thread.stop()
            rec = self.running_recording
            self.running_recording = None
            self.running_thread = None
            return rec

    def stop_rec(self):
        """Stops the current recording"""
        rec = self._stop_rec()
        self.recordings.append(rec)
        if self.stream is not None:
            self.stream.send(rec)
        return self

    def stop_rec_to_stream(self, stream=None):
        """Stops the current recording and calls the given stream's send method with the recording
        @params:
        stream: Overrides the classes default stream for this function call
        """
        stream = stream or self.stream
        assert stream is not None, "You haven't set a stream yet"
        rec = self._stop_rec()
        stream.send(rec)
        return self

    def abort_rec(self):
        """Aborts the recording (Stops the recording without saving it)"""
        self._stop_rec()
        return self

    def get_recording(self):
        """
        This gets the first recording in the list of recordings.
        In many cases you just have one recording and then you get that one
        """
        assert len(self.recordings) > 0, "You haven't finished a recording yet"
        return self.recordings[0]

    def get_recordings(self):
        """
        Get all recordings until now
        """
        return self.recordings
    
    def save_recording(self, filename: str):
        """Saves a single recording given a filename. 
        @params:
        filename: str
            A filename to save to. Available extensions are in the documentation or by calling `available_formats()`
        """
        return self.get_recording().save(filename)

    # @timer_wrapper
    def save_recordings(
        self,
        key: Union[str, Sequence[str], Callable[[int], Optional[str]]],
        save_dir: Optional[AnyPath] = None,
        blocking: bool = True,
    ):
        """Inits a RecordingSaver with the recordings and calls its save function with the given arguments.
        @params:
        key (str | Sequence[tuple[str,str]] | Callable[[int],tuple[str,str]]): If the key is a str then it is the format in which the recordings are saved.
        Otherwise the key will be given the index of the recording and should return a valid (filename,extension) tuple for each recording.
        If the key instead returns None the recording will be skipped.
        save_dir (AnyPath): The directory where to save the recordings. Takes anything that os.chdir takes. Defaults to ""
        blocking (bool): Whether to block the calling process until the recordings are all saved or not. Easiest and default is True
        @returns
        If the RecordingSaver was set to blocking then this returns a list of filepaths.
        Otherwise it returns a function that returns that list and **has** to be called before the program ends.
        """
        if not self.recordings:
            logging.debug("No recordings to save")
        saver = RecordingSaver(self.recordings, key, save_dir, blocking)
        return saver.save()


##############################################################################################################


##############################################################################################################
# Saving
# https://stackoverflow.com/a/55596396/15046005
codec_dict = {"avi": "DIVX", "webm": "WEBM", "mp4": "h264"}

def fourcc(*args: str)->int:
    return cv2.VideoWriter_fourcc(*args) # type: ignore

try:
    _codec_dict = {key: fourcc(*v) for key, v in codec_dict.items()}
except:
    logging.info(
        "ffmpeg is not available. Install it to save your recording as a video"
    )

def add_codec(format: str, codec: Union[int, str]):
    """
    Adds a codec to the codec_dict to allow saving it via ffmpeg
    """
    _codec_dict[format] = fourcc(*codec) if isinstance(codec, str) else codec


def available_formats():
    """ The available formats to save a recording"""
    return [*_codec_dict.keys(), "npz"]


@contextmanager
def _video_writer(path: str, codec: int, fps: float, dimensions: Tuple[int, int]):
    """Internal contextmanager that automatically handles the opening and closing of a cv2.VideoWriter"""
    video = cv2.VideoWriter(os.path.abspath(path), codec, fps, dimensions)
    try:
        yield video
    finally:
        video.release()


def _save_single(
    frames: List[np.ndarray], fps: float, size: Tuple[int, int], filename: str
):
    ext = os.path.splitext(filename)[1][1:]
    if ext == "npz":
        return _save_as_np(frames, fps, size, filename)
    elif codec_dict.get(ext):
        return _save_as_video(frames, fps, size, filename, ext)
    raise ValueError("Extension not supported: " + ext)


def _save_as_video(
    frames: List[np.ndarray], fps: float, size: Tuple[int, int], path: str, ext: str
):
    try:
        codec = _codec_dict[ext]
        with _video_writer(path, codec, fps, size) as video:
            for frame in frames:
                video.write(_pg_to_cv2(
                    frame
                ))
    except NameError as e:
        if e.name == "_codec_dict":
            raise RuntimeError("ffmpeg isn't available.")
        else:
            raise e


def _save_as_np(frames: List[np.ndarray], fps: float, size: Tuple[int, int], path: str):
    meta = np.array([fps, *size], dtype=np.float64)
    np.savez_compressed(path, *frames, meta=meta)


class RecordingSaver:
    """A class that is responsible for saving the recordings"""
    key: Callable[[int], Optional[str]]
    def __init__(
        self,
        recordings: List[Recording],
        key: Union[str, Sequence[str], Callable[[int], Optional[str]]],
        save_dir: Optional[AnyPath] = None,
        blocking: bool = True,
    ):
        """Inits a RecordingSaver

        Parameters
        ----------
        recordings: List[Recording]
            The list of recordings to save.
        key: str | Sequence[str] | Callable[int, Optional[str]]
            If the key is a str then it is the format in which the recordings are saved.
            Otherwise the key will be given the index of the recording and should return a valid filename for each recording.
            If the key instead returns None the recording will be skipped.
        save_dir: AnyPath
            The directory where to save the recordings. Takes anything that os.chdir takes. If not given or set to None then saves in the current directory ""
        blocking: bool
            Whether to block the calling process until the recordings are all saved or not. Easiest and default is True
        """
        try:
            if isinstance(key, str):

                def str_lambda(x: int):
                    return f"recording_{x}.{key}"

                self._key = str_lambda
            elif callable(key):
                self._key = key # type: ignore # We trust the user to give a valid callable
            else:
                self._key = lambda x: key[x]
        except NameError as e:
            if e.name == "__getitem__":
                raise TypeError(
                    "key has to be one of str, callable or Sequence")
            else:
                raise
        self.recordings = recordings
        self.save_dir = save_dir or "."
        self.blocking = blocking
        self.saved_recordings: List[str] = []
        assert os.path.isdir(self.save_dir), f"save_dir has to be a directory: " + str(
            save_dir
        )

    def _save(self)->None:
        """Thread target function for saving"""
        assert self.recordings is not None, "recordings have to be set before saving"
        if self.save_dir is not None:
            old_wd = os.getcwd()
            os.chdir(self.save_dir)
        threads = []
        for i, rec in enumerate(self.recordings):
            filename: Optional[str] = self.key(i)
            if filename is not None:
                thread = Thread(target=rec.save, args=filename)
                thread.start()
                threads.append((thread, filename))
        for thread, saved_rec in threads:
            thread.join()
            self.saved_recordings.append(saved_rec)
        if self.save_dir is not None:
            os.chdir(old_wd)

    def save(self) -> Union[Callable[[], List[str]], List[str]]:
        """Saves the given recordings.

        If the RecordingSaver was set to blocking then this returns a list of paths.
        Otherwise it returns a function that returns that list and **has** to be called before the program ends.
        """
        try:
            save_thread = Thread(target=self._save, daemon=(not self.blocking))
        except:
            raise RuntimeError("Save thread failed")
        save_thread.start()
        if self.blocking:
            save_thread.join()
            return self.saved_recordings
        else:

            def func():
                save_thread.join()
                return self.saved_recordings

            return func


##############################################################################################################


##############################################################################################################
# Playing
def play_indefinite(player: "RecordingPlayer"):
    player.restart()


class play_n_times:
    def __init__(self, n: int):
        self.n = n

    def __call__(self, player: "RecordingPlayer"):
        self.n -= 1
        if self.n >= 1:
            player.restart()


class play_n_wrapper:
    def __init__(self, n: int):
        self.n = n
        self.func = None

    def __call__(self, func):
        @wraps(func)
        def on_stop(player: "RecordingPlayer"):
            try:
                res = func(player)
            except TypeError:
                res = func()
            self.n -= 1
            if res and self.n >= 1:
                player.restart()

        return on_stop


def npz_player(path: AnyPath, *args, **kwargs):
    """In most cases use `NPZPlayer`"""
    npzFile = np.load(path)
    fps, w, h = npzFile["meta"]
    size = int(w), int(h)
    rec = Recording(fps, size, compress=0)
    for i in range(len(npzFile.files)):
        rec.add_frame(pg.surfarray.make_surface(npzFile[f"arr_{i}"]))
    return RecordingPlayer(rec, *args, **kwargs)


class RecordingPlayer:
    """
    A RecordingPlayer that takes your pure recordings and plays them very easily.
    """

    def __init__(
        self,
        rec: Recording,
        on_stop: Optional[Callable[[], None]] = None,
        surf: Optional[pg.surface.Surface] = None,
    ):
        self.rec = rec
        self.on_stop = on_stop if on_stop is not None else mt
        self.surf = surf or pg.display.get_surface()
        self.rec.resize(self.surf.get_size())
        self._init_thread()

    def _on_stop(self):
        try:
            self.on_stop(self)
        except TypeError:
            self.on_stop()

    def _init_thread(self):
        """Inits a new thread for the player"""
        if hasattr(self, "thread"):
            self.thread.stop()
        self.thread = PlayingThread(
            self.surf, self.rec.dt, self._get_frame, self._on_stop
        )

    def _get_frame(self, pos: int):
        frame = self.rec.frames[pos]
        frame = decompress(frame, self.rec.compress)
        return frame

    def stop(self):
        """Forcefully stop a player even before he ends naturally. Can be called several times."""
        self.on_stop = mt
        self.thread.stop()
        self.thread.join()
        return self

    def play(self):
        """Plays the player. Can be called multiple times. If the player was paused previously, it will resume."""
        if self.is_("stopped"):
            raise RuntimeError("""Restart a finished player with restart""")
        elif self.is_("paused"):
            self.thread.play()
        elif not self.is_("started"):
            self.thread.start()
        return self

    def pause(self):
        """Pauses the player. Can be called multiple times.
        Raises a RuntimeError if the player hasn't started yet or already finished"""
        if self.thread.is_alive():
            self.thread.pause()
            return self
        else:
            if self.thread.running:
                raise RuntimeError("You cannot pause before starting to play")
            else:
                raise RuntimeError(
                    "The player already finished. You cannot pause anymore. Restart a finished player with restart"
                )

    def restart(self):
        """Starts playing from the beginning, whether the player stopped or not"""
        print("Restarting")
        if self.is_("stopped"):
            self._init_thread()
            self.thread.start()
        else:
            self.thread.position = 0
            self.thread.play()
        return self

    def seek(self, position: int):
        """
        Sets the current position of the player. Can be called before the player started.
        This won't catch if the position is too high.
        Instead the player will just finish on the next iteration.
        Raises an TypeError if the position is not an int.
        """
        try:
            self.thread.position = int(position)
            return self
        except TypeError:
            raise TypeError(
                f"Position (type:{position.__class__.__name__} couldn't be converted to an integer"
            )

    def tell(self) -> int:
        """Returns the current position of the player"""
        return self.thread.position

    def seekms(self, time: float):
        """Goes to the specified time in ms (floor)"""
        return self.seek(int(time / self.rec.dt))

    def tellms(self) -> float:
        """Convinience function: returns the current play time. `self.rec.dt * self.tell()`"""
        return self.rec.dt * self.tell()

    def is_(self, attr: str) -> bool:
        """Test the state of the player

        Parameters
        ----------
        attr (str): select what state you want to get.
            - "started" -> whether the player is started (and still running)
            - "stopped" -> whether the player has stopped/finished
            - "playing" -> whether the player is playing
            - "paused" -> whether the player is paused

        Raises
        ------
        TypeError:
            If attr is not of type str
        ValueError:
            If attr is none of the above
        """
        ensure_type(attr, str)
        if attr == "started":
            return self.thread.is_alive() and self.thread.running
        elif attr == "stopped":
            return not self.thread.running
        elif attr == "playing":
            return self.thread.is_alive() and not self.thread.paused
        elif attr == "paused":
            return self.thread.is_alive() and self.thread.paused
        else:
            raise ValueError(
                f'attr must be either {" or ".join(["started","stopped","playing","paused"])}'
            )


class NPZPlayer(RecordingPlayer):
    def __init__(
        self,
        path: AnyPath,
        on_stop: Optional[Callable[[], None]] = None,
        surf: Optional[pg.surface.Surface] = None,
    ):
        """Inits a NPZPlayer

        Parameters
        ----------
        path: AnyPath
            The file to be opened
        on_stop: Optional[Callable[[],None]]
            What to do when the Player finished
        surf: Optional[pg.surface.Surface]
            The Surface to play on
        """
        self.npzFile = np.load(path)
        self.surf = surf or pg.display.get_surface()
        self.on_stop = on_stop or mt
        fps, w, h = self.npzFile["meta"]
        self.fps = fps
        self.size = int(w), int(h)
        self._init_thread()

    def _init_thread(self):
        self.thread = PlayingThread(
            self.surf, 1000 / self.fps, self._get_frames, self._on_stop
        )

    def _get_frames(self, position: int) -> pg.surface.Surface:
        try:
            np_arr = self.npzFile[f"arr_{position}"]
            frame = pg.surfarray.make_surface(np_arr)
            return frame
        except KeyError:
            raise IndexError


##############################################################################################################

atexit.register(cleanup)
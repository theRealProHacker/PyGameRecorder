from collections import defaultdict
import json
import logging
import random
import time
from typing import Any, Callable, Iterable, Optional, Union, List, Dict
from dataclasses import dataclass, astuple
import pygame as pg

################################################################################################
# helpers


def skip_none(x, y):
    return y if x is None else x


def dfilter(
    d: dict,
    *,
    key: Optional[Union[Callable[[Any], bool], list]] = None,
    value: Optional[Union[Callable[[Any], bool], list]] = None,
):
    """Filter a dictionary by key or value. You must specify either key or value"""
    if (key is None) ^ (value is None):  # xor
        if value is None:
            if isinstance(key, list):
                key = key.__contains__
            return {k: v for k, v in d.items() if key(k)}  # type: ignore
        elif key is None:
            if isinstance(value, list):
                value = value.__contains__
            return {k: v for k, v in d.items() if value(v)}
    else:
        raise TypeError("Exactly one of key and value have to be None")


start_lambda = lambda starts: lambda k: any((k.startswith(s) for s in starts))


def starts_with(s: Union[Iterable[str], str]):
    """Returns all the pg.locals that start with any of the given strings"""
    try:
        return dfilter(pg.__dict__, key=start_lambda(s)).values()
    except TypeError:
        return dfilter(pg.__dict__, key=start_lambda([s])).values()


time_since = lambda t: time.time() - t


def false_lambda():
    return False


def dict_lambda():
    return {}


def read(path: str):
    with open(path, "r") as f:
        return json.load(f)


def write(object: Any, path: str):
    with open(path, "w") as f:
        json.dump(object, f, ensure_ascii=False)


def find(pred: Callable[[Any], bool], l: Iterable) -> Any:
    for x in l:
        if pred(x):
            return x


def print_names(l: Union[Iterable[pg.event.Event], Iterable[int]]):
    try:
        for e in l:
            print(pg.event.event_name(e.type))  # type: ignore
    except (AttributeError, TypeError):
        for e in l:
            print(pg.event.event_name(e))  # type: ignore


def safe_item_set(d: dict):
    conv_l_t = lambda l: tuple(l) if type(l) is list else l
    return {(k, conv_l_t(v)) for k, v in d.items()}


################################################################################################


################################################################################################
# Typing
@dataclass
class SavedEvent:
    type: int
    attrs: dict
    time: float


################################################################################################


################################################################################################
# compressions
defaults: Dict["str", Dict["str", Any]] = {
    "window": {
        "value": None,
        "events": [
            32774,
            32785,
            770,
            32776,
            32783,
            1024,
            1025,
            1026,
            768,
            771,
            769,
            32784,
            32786,
            32777,
            32787,
        ],
    },
    "touch": {"value": False, "events": [1024]},
    "buttons": {"value": [0] * 3, "events": [1024]},  #  mouse move
    "mod": {"value": 0, "events": [771, 769]},  # Key up/key down
}


def_lookup: Dict[int, Dict[str, Any]] = defaultdict(dict)

for attr, d in defaults.items():
    for event in d["events"]:
        def_lookup[event][attr] = d["value"]


def single_compress(se: SavedEvent) -> SavedEvent:
    """
    Mutates the given SavedEvent and returns it
    """
    if (default := def_lookup.get(se.type)) is not None:
        eit = safe_item_set(se.attrs)
        se.attrs = {k: v for k, v in (eit - (eit & safe_item_set(default)))}
    return se


def single_decompress(se: SavedEvent):
    """
    Mutates the given SavedEvent and returns it
    """
    if (d := def_lookup.get(se.type)) is not None:
        se.attrs |= d
    return se


################################################################################################


################################################################################################

old_events = {
    "mouse_foc": pg.mouse.get_focused,
    "mouse_pos": pg.mouse.get_pos,
    "mouse_rel": pg.mouse.get_rel,
    "mouse_pressed": pg.mouse.get_pressed,
    "key_foc": pg.key.get_focused,
    "key_pressed": pg.key.get_pressed,
    "key_mods": pg.key.get_mods,
}


class EventRegister:
    """
    As the name suggests this class registers the pygame events so that you can then save them.
    """

    starts = {"MOUSE", "KEY", "WINDOW"}

    def __init__(
        self,
        in_out: str = "in",
        path: Optional[str] = None,
        events: Optional[Iterable[int]] = None,
        no_catch_events: Optional[Iterable[int]] = None,
    ):
        default_events = set(starts_with(self.starts))
        self.events: set[int] = set(skip_none(events, default_events)) | {pg.QUIT}
        self.no_catch_events: set[int] = set(skip_none(no_catch_events, set())) | {
            pg.QUIT
        }
        assert in_out in ("in", "out"), 'in_out has to be either "in" or "out"'
        self._type = in_out
        self.path: Optional[str] = path
        self.random_seed = None
        self.start = time.time()
        self.rec_events: list[SavedEvent] = []
        self._reci = 0
        if self._type == "out":
            assert isinstance(self.path, str), "path must be a string"
            r = read(self.path)
            self.random_seed = r["random_seed"]
            self.rec_events = [single_decompress(SavedEvent(*x)) for x in r["events"]]
            self._focus = pg.mouse.get_focused()
            self._mouse_pos = pg.mouse.get_pos()
            self._mouse_rel = pg.mouse.get_rel()
            self._mouse_pressed = list(pg.mouse.get_pressed())
            self._key_pressed: dict = defaultdict(false_lambda)
            for i, pressed in enumerate(pg.key.get_pressed()):
                self._key_pressed[i] = pressed
            self._key_mod = pg.key.get_mods()
            pg.mouse.get_focused = self.focused
            pg.mouse.get_pos = self.mouse_pos
            pg.mouse.get_rel = self.mouse_rel
            pg.mouse.get_pressed = self.mouse_pressed  # type: ignore
            pg.key.get_focused = self.focused
            pg.key.get_pressed = self.key_pressed
            pg.key.get_mods = self.key_mods

    def _add_events(self, events: Iterable[pg.event.Event]):
        assert self._type == "in", "Can only add events when inputting"
        for e in events:
            if (t := e.type) in self.events:  # python 3.10
                self.rec_events.append(
                    SavedEvent(t, e.__dict__, time_since(self.start))
                )

    def _get_events(self) -> List[pg.event.Event]:
        assert self._type == "out", "Can only get events when outputting"
        this_events: list[pg.event.Event] = []
        for e in self.rec_events[self._reci :]:
            if time_since(self.start) >= e.time:
                self._reci += 1
                this_events.append(pg.event.Event(e.type, e.attrs))
                if e.type == pg.MOUSEMOTION:
                    self._mouse_pos = e.attrs["pos"]
                    self._mouse_rel = e.attrs["rel"]
                elif e.type == pg.MOUSEBUTTONDOWN:
                    try:
                        self._mouse_pressed[e.attrs["button"] - 1] = True
                    except IndexError:
                        pass
                elif e.type == pg.MOUSEBUTTONUP:
                    try:
                        self._mouse_pressed[e.attrs["button"] - 1] = False
                    except IndexError:
                        pass
                elif e.type in (pg.KEYDOWN, pg.KEYUP):
                    self._key_mod = e.attrs["mod"]
                    if e.type == pg.KEYDOWN:
                        self._key_pressed[e.attrs["key"] - self._key_mod] = True
                    elif e.type == pg.KEYUP:
                        self._key_pressed[e.attrs["key"] - self._key_mod] = False
                elif e.type == pg.WINDOWFOCUSGAINED:
                    self._focus = True
                elif e.type == pg.WINDOWFOCUSLOST:
                    self._focus = False
            else:
                return this_events
        return this_events

    def seed(
        self, random_seed: Union[int, float, str, bytes, bytearray] = 0
    ) -> "EventRegister":
        """
        Seed the `EventRegister` a random seed.
        Just call this on the instance to have a work around for random games (ie. most games).
        If you pass a non-faulty `random_seed` to this then it will overwrite the instances current `random_seed`.
        But in most cases just don't pass anything.
        """
        if self._type == "in":
            self.random_seed = random_seed or random.random()
        else:
            self.random_seed = random_seed or self.random_seed
        random.seed(self.random_seed)
        return self

    def get_events(self) -> List[pg.event.Event]:
        """Get the next events. To use an `EventRegister`, call this instead of `pg.event.get()`"""
        events = pg.event.get()
        if self._type == "out":
            return [
                *filter(lambda e: e.type in self.no_catch_events, events)
            ] + self._get_events()
        elif self._type == "in":
            self._add_events(events)
            return events
        else:
            raise RuntimeError('self._type must be "out" or "in"')

    def save(self):
        """Wraps up the `EventRegister`. You must call this at the end of your script."""
        if self._type == "in":
            if not self.rec_events[-1].type == pg.QUIT:
                logging.debug("Saving but PyGame hasn't finished yet")
                self.rec_events.append((pg.QUIT, {}, self.rec_events[-1][2] + 0.1))
            events = [astuple(single_compress(x)) for x in self.rec_events]
            write({"events": events, "random_seed": self.random_seed}, self.path)
        else:
            pg.mouse.get_focused = old_events["mouse_foc"]
            pg.mouse.get_pos = old_events["mouse_pos"]
            pg.mouse.get_rel = old_events["mouse_rel"]
            pg.mouse.get_pressed = old_events["mouse_pressed"]
            pg.key.get_focused = old_events["key_foc"]
            pg.key.get_pressed = old_events["key_pressed"]
            pg.key.get_mods = old_events["key_mods"]

    # class getters

    def focused(self):
        return self._focus

    def mouse_pos(self):
        return self._mouse_pos

    def mouse_rel(self):
        return self._mouse_rel

    def mouse_pressed(self):
        return tuple(self._mouse_pressed)

    def key_pressed(self):
        return self._key_pressed

    def key_mods(self):
        return self._key_mod


################################################################################################


__all__ = ["EventRegister"]

################################################################################################

if __name__ == "__main__":
    print("EventRegister")

"""
Regression test for the plural save path (`ScreenRecorder.save_recordings` /
`RecordingSaver`). Both were silently broken before 0.1.2:
  - `RecordingSaver._save` referenced `self.key` instead of `self._key`
  - the per-recording save thread was started with `args=filename` (a str),
    which unpacks into one positional arg per character instead of one filename

We use the `npz` format so the test does not require ffmpeg.
"""

import os

# Headless pygame so the test runs in CI without a display
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame as pg

from pygame_screen_record import RecordingSaver, ScreenRecorder


def _make_recorder():
    pg.init()
    pg.display.set_mode((32, 24))
    recorder = ScreenRecorder(30)
    recorder.start_rec()
    # let at least one frame be captured
    for _ in range(3):
        pg.time.wait(20)
    recorder.stop_rec()
    return recorder


def test_save_recordings_npz(tmp_path):
    recorder = _make_recorder()
    paths = recorder.save_recordings("npz", str(tmp_path))
    assert paths == ["recording_0.npz"]
    assert (tmp_path / "recording_0.npz").is_file()


def test_recording_saver_with_sequence_key(tmp_path):
    recorder = _make_recorder()
    recordings = recorder.get_recordings()
    saver = RecordingSaver(recordings, ["only_one.npz"], str(tmp_path))
    paths = saver.save()
    assert paths == ["only_one.npz"]
    assert (tmp_path / "only_one.npz").is_file()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_save_recordings_npz(Path(tempfile.mkdtemp()))
    test_recording_saver_with_sequence_key(Path(tempfile.mkdtemp()))
    print("save tests passed")

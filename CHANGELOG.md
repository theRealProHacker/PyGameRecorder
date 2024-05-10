Uses [Semantic Versioning](http://semver.org/)

# 0.1.1

A hotfix addressing an issue in the README and a few more chaining opportunities.

# 0.1.0

1. Replaced the (filename, extension) tuple to a single filename string that includes the extension for simplicity. The extension is extracted using `os.path.splitext()`
2. Renamed `ScreenRecorder.get_single_recording()` to `ScreenRecorder.get_recording` and added `ScreenRecorder.save_recording()`
3. Removed `PlayerList` completely

Also, the README was changed a lot (hopefully for the better) and I fixed some of the typing using `mypy`. 

# 0.0.6

Fixed a bug in `README.md` 

# 0.0.4

1. Fixed a simple bug where a nonexistent method was called. 
2. Made imports easier to do directly from `pygame_screen_record`

# 0.0.3

Added support for 3.8
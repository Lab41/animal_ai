[Link to original README](README_OLD.md)


# Running AnimalAI remotely
The AnimalAI environment requires rendering to a screen. There are several ways of virtually emulating a screen in Linux. (This is not the same as the `screen` command.)

## Method 1

## Method 2
The original documentation has a page for [training in the cloud](documentation/cloudTraining.md). This method uses `xvfb` (X virtual frame buffer) which renders the output in virtual memory so it is never actually available for viewing. I briefly tried this method but didn't get it to work. You must set `dockerTraining=True` in your Python code in order to activate the xvfb wrapper. See [these lines of code](animalai/animalai/envs/environment.py#L201) for more
info.
You could also try running like below: 
`xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' python trainDopamine.py`
or 
```
Xvfb :5 -screen 0 800x600x24 &
export DISPLAY=:5
python trainDopamine.py 
```
This will most likely produce an error like below: 
```
ljt@shrike:~$ cat /home/ljt/.config/unity3d/Unity\ Technologies/Unity\ Environment/Player.log
Desktop is 800 x 600 @ 0 Hz
Unable to find a supported OpenGL core profile
Failed to create valid graphics context: please ensure you meet the minimum requirements
E.g. OpenGL core profile 3.2 or later for OpenGL Core renderer
Vulkan detection: 0
No supported renderers found, exiting

(Filename:  Line: 634)
```
which didn't make sense to me given: 
```
(animal_ai_private) ljt@shrike:~/animal_ai_private$ glxinfo | grep core
    Preferred profile: core (0x1)
    Max core profile version: 3.3
OpenGL core profile version string: 3.3 (Core Profile) Mesa 18.2.2
OpenGL core profile shading language version string: 3.30
OpenGL core profile context flags: (none)
OpenGL core profile profile mask: core profile
OpenGL core profile extensions:
```

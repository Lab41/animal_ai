[Link to original README](README_OLD.md)


# Running AnimalAI remotely
The AnimalAI environment requires rendering to a screen (X windows system). There are several ways of virtually emulating a screen in Linux. (This is not the same as the `screen` command.)

## Method 0 (requires physical screen/X window)
This method will connect to the physical display and render the graphics there. You will not see anything on your remote computer. 
```
ssh host
export DISPLAY=:1  # or :0 depending on your version of Linux
# run your scripts as usual
```

## Method 1

### DCOS instructions
Add Networking Service endpoints when creating a new instance. 
My endpoints are listed below: 
1. Container Port: 8888, Host Port: Assign Automatically; Used for Jupyter 
2. Container Port: 22, Host Port: Assign Automatically; Used for SSH, you will need to make note of which port is assigned and use that for ssh'ing to your instance. 
3. Container Port: 5900+n (where n is the display number used below in setting up TurboVNC), Host Port: 5900+n; Used for TurboVNC



Install VirtualGL and TurboVNC (based on instructions from [here](https://gist.github.com/cyberang3l/422a77a47bdc15a0824d5cca47e64ba2)

The install files for Ubuntu are in the [remote](remote) directory. 
Other versions can be download at [TurboVNC Download](https://sourceforge.net/projects/turbovnc/) and [VirtualGL Download](https://sourceforge.net/projects/virtualgl/)

```
sudo dpkg -i ./remote/virtualgl_*.deb
sudo /opt/VirtualGL/bin/vglserver_config
sudo dpkg -i ./remote/turbovnc_*.deb

# run the line below to only allow one time password authentication (I thought this had the best trade off between security and ease of use)
# sudo bash -c 'echo "permitted-security-types = TLSOtp" >> /etc/turbovncserver-security.conf'

echo "PATH=\$PATH:/opt/TurboVNC/bin" >> ~/.bashrc
```
Change n to some number in the following commands. n will be your display number. 
To start running a server on display n with one time password (otp) enabled: 
```
vncserver -otp :n
```
The VNC server will now be running on port 5900+n

To retrieve a one time password, first connect to your system then run: 
```
vncpasswd -o -display :n
```

I added the following alias to my laptop in order to easily connect: 
```
alias vnc='/opt/TurboVNC/bin/vncviewer 10.225.137.15:2 -Password `ssh desktop "vncpasswd -o -display :2 2>&1 | sed -e '"'"'s/.*://'"'"'"`'
```


## Method 2 (requires physical screen)
Install Teamviewer. 
This has some lag but probably the easiest setup. 

## Method 3
Forward your X session over ssh. There are many online tutorials for this. This method is similar to Method 1.

## Method 4
The original documentation has a page for [training in the cloud](documentation/cloudTraining.md). This method uses `xvfb` (X virtual frame buffer) which renders the output in virtual memory so it is never actually available for viewing. You must set `dockerTraining=True` in your Python code in order to activate the xvfb wrapper. See [these lines of code](animalai/animalai/envs/environment.py#L201) for more
info. I briefly tried this method but kept getting OpenGL errors so I stopped trying. 

Here are some other things I tried/learned if you want to use this method.

You could also try running:
```
xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' python trainDopamine.py
```
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


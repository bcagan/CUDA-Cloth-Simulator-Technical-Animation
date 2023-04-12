## Overview

This source code provides a template to implement cloth simulation. Please refer to the [course website](http://graphics.cs.cmu.edu/nsp/course/15464-s20/www/assts.html) for project instructions.

## Controls

The application reads data from a directory tree, expecting one asf file and
possibly several amc files per directory. Once the motion has been loaded,
a

* ESC to quit

* mouse buttons for camera control. One the Mac, the three mouse
buttons are obtained using the normal button, the normal button while
holding down the option key, and the normal button while holding down
the "Apple" key.

* 'r' will reset cloth simulation

## Compiling

### 1. Windows

The environment has been setup for windows under VS2017. Click ide/VS2017/amc_viewer.sln and run.

### 2. Ubuntu 18.04

Step 1. Install dependencies (sdl1.2.15-dev, libxml2-dev) and compiling tools (jam)
```
sudo apt-get install alien
mkdir tmp
cd tmp
wget https://www.libsdl.org/release/SDL-devel-1.2.15-1.x86_64.rpm
sudo alien -i SDL-devel-1.2.15-1.x86_64.rpm
cd ..
rm -r tmp
sudo apt-get install libxml2-dev
sudo apt-get install jam
```

Step 2. Compile with jam
```
jam
```

Step 3. Test
```
cd Dist
./browser
```

## Copyright
All source is copyright Jim McCann unless otherwise noted. Feel free to use
in your own projects (please retain some pointer or reference to the original
source in a readme or credits section, however).

Contributing back bugfixes and improvements is polite and encouraged but not
required.

## Contibutors
This AMCViewer is originally written by Jim McCann (jmccann@cs.cmu.edu). 

Here's a list of people who may also have contributed: \
Roger Dannenberg (rbd@cs.cmu.edu) \
Se-Joon Chung (sejoonc@cs.cmu.edu) \
Yanzhe Yang (yanzhey@cs.cmu.edu)

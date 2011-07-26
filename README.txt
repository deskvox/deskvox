Welcome to DeskVOX Volume Explorer (DeskVOX)

0. Content
============================================
1. How to obtain DeskVOX

2. Additional dependencies

3. How to build deskvox and vconv from source

4. How to build the virvo library from source
============================================

1. How to obtain DeskVOX

Download a zip archive from:

http://sourceforge.net/projects/deskvox/

Check out DeskVOX via Subversion:

svn co https://deskvox.svn.sourceforge.net/svnroot/deskvox deskvox
============================================

2. Additional dependencies

The virvo library won't build without the following libraries installed

GLEW: The OpenGL Extension Wrangler Library

The DeskVOX applicaton won't build without the following libraries installed

FOX TOOLKIT 1.6

Having the following libraries installed is recommended but not necessary

Cg Toolkit (NVIDIA COORP)
Cuda Toolkit (NVIDIA COORP)
============================================

3. How to build deskvox and vconv from source

DeskVOX uses the CMake build system to generate a project specific to your
platform to be built from. Obtain CMake from http://www.cmake.org or by means
provided by your Operating System.

Switch to the topmost folder of the deskvox package you downloaded or the
topmost folder of the branch you wish to build if you checked out DeskVOX via
Subversion (i.e. the folder this document is located in).

CMake encourages you to do so called out-of-source builds, i.e. all files, e.g.
object files, executables or auto-generated headers will be located in a folder
separate from the folder the source files are located in.
In order to perform an out-of-source build, create a new build directory, e.g.:

$ mkdir build

Change to that directory:

$ cd build

Envoke CMake:

$ cmake ..

This will generate a build environment specific to your platform, e.g. Visual
Studio Solutions on Windows or a Makefile project on Unix.
Edit CMakeCache.txt to specify custom paths to additional libraries, perform
a Debug build, etc. .

On Unix platforms, type:

$ make

The static virvo library will be located in ${BUILD_DIR}/virvo/virvo
The vview test application will be located in ${BUILD_DIR}/virvo/test
The deskvox application will be located in ${BUILD_DIR}/vox-desk
The vconv application will be located in ${BUILD_DIR}vox-desk

In order to install deskvox and its associated files, on Unix type:

$ make install

Deskvox will be installed to a default location, which can be modified
before installing by editing CMakeCache.txt in ${BUILD_DIR}
============================================

4. Hot to build the virvo library from source

In order to build only the virvo library, switch to the virvo folder right
under the topmost folder of the one this document is located in. As
described under 3., create a build folder for an out-of-source build, e.g.:

$ mkdir build

Change to that directory:

$ cd build

Envoke CMake:

$cmake ..

Optionally edit CMakeCache.txt (for a more detailled description, see 3.)

On Unix platforms, type:

$ make

The virvo library will be located in ${BUILD_DIR}/virvo
============================================

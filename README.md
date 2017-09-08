afx-nuke-plugins﻿
================
## Requirements
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [Intel IPP](https://registrationcenter.intel.com/en/forms/?productid=2558) (Free Community Licensing)
* [BOOST](https://sourceforge.net/projects/boost/files/boost/)
* Nuke 10.0v1+ NDK (Installed with Nuke)

#### Optional
* [Hoard](https://github.com/emeryberger/Hoard/releases)
* [IlmBase](http://www.openexr.com/downloads.html)

## Build and Install
### Linux
*"Plug-ins compiled with GCC versions 4.1 through 4.8.2 without C++11 support should be compatible...The use of C++11 via GCC, clang or the Intel compiler is untested and unsupported, especially in terms of passing standard library objects targeted at C++11 through the plug-in interface."*
[(NDK Dev Guide)](https://www.thefoundry.co.uk/products/nuke/developers/105/ndkdevguide/appendixa/linux.html)

These plugins use some C++11 language features and therefore requires GCC 4.8. GCC 4.8.1 was the first feature-complete implementation of the 2011 C++11 standard.  Although The Foundry states that the use of C++11 is untested and unsupported, the usage of C++11 in afx-nuke-plugins has not caused any issues in testing. GCC 4.8.2 is the recommended version.

To check the system version of GCC:
```bash
gcc --version
```

GCC 4.8.2:

dependencies: libgmp-dev libmpc-dev 

* Download [GCC 4.8.2 source.](https://gcc.gnu.org/mirrors.html)
```bash
tar -xvzf gcc-4.8.2.tar.gz
cd gcc-4.8.2
mkdir build
cd build
../configure --prefix=/usr/local/gcc-4.8.2 --disable-multilib --enable-language=c,c++
make -j$(nproc) bootstrap
make -j$(nproc)
sudo make install
```

Boost:

* Download [boost source](http://www.boost.org/users/download/)
* Extract and cd into boost root
```bash
export BOOST_VERSION=${PWD##*/}
cd tools/build
./bootstrap.sh
./b2 install --prefix=../../boost.build
cd ../..
echo "using gcc : 4.8.2 : /usr/local/gcc-4.8.2/bin/g++ : root=/usr/local/gcc-4.8.2 : <cxxflags>-std=c++11 ;" >> boost.build/user-config.jam
export BOOST_BUILD_PATH=$(pwd)/boost.build
sudo -E boost.build/bin/b2 --prefix=/usr/local/${BOOST_VERSION} --build-dir=$(pwd)/build --with-thread toolset=gcc-4.8.2 variant=release link=shared threading=multi runtime-link=shared install
sudo ln -s /usr/local/${BOOST_VERSION} /usr/local/boost
```

afx-nuke-plugins:

```bash
git clone https://github.com/AuthorityFX/afx-nuke-plugins.git
cd afx-nuke-plugins
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=/usr/local/gcc-4.8.2/bin/gcc -DCMAKE_CXX_COMPILER=/usr/local/gcc-4.8.2/bin/g++ ..
make -j$(nproc) install
```

### Windows
*"NUKE on Windows is compatible with plug-ins built with: Visual Studio 2010 - (Manifest Version: 10.0.30319)*"
[(NDK Dev Guide)](https://www.thefoundry.co.uk/products/nuke/developers/105/ndkdevguide/appendixa/windows.html)

Download and install the following build tools:
* [CMake](https://cmake.org/download)
* [Git](https://git-scm.com/downloads)
*  [Windows 7 SDK and .NET Framework 4](https://www.microsoft.com/en-us/download/details.aspx?id=8279) or [Visual Studio 2010 Ultimate](http://download.microsoft.com/download/4/0/6/4067968E-5530-4A08-B8EC-17D2B3F02C35/vs_ultimateweb.exe)

##### With Windows 7 SDK:
Launch Cmd.exe
```dos
C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.Cmd" /Release /x64
git clone https://github.com/AuthorityFX/afx-nuke-plugins.git
cd afx-nuke-plugins
mkdir build
cd build
cmake -G "Visual Studio 10 2010 Win64" ..
cmake --build . --target INSTALL --config Release
```
##### With Visual Studio 2010:
Launch Cmd.exe
```dos
git clone https://github.com/AuthorityFX/afx-nuke-plugins.git
cd afx-nuke-plugins
mkdir build
cd build
cmake -G "Visual Studio 10 2010 Win64" ..
cmake --build . --target INSTALL --config Release
```
##### Potential Problems
If .NET Framework 4.x and/or Visual Studio C++ 2010 Redistributable are installed:
* Uninstall conflicting packages
* Install Windows 7 SDK
* Reinstall confliting packages

Installing Visual Studio 2010 Service Pack 1 may remove compilers and libraries installed by the Windows SDK. The following solution is from [MSDN blog](https://blogs.msdn.microsoft.com/vcblog/2011/03/31/released-visual-c-2010-sp1-compiler-update-for-the-windows-sdk-7-1/)

Installation order:
1. Visual Studio 2010
2. Windows SDK 7.1
3. Visual Studio SP1
4. [Visual C++ 2010 SP1 Compiler Update for the Windows SDK 7.1](https://www.microsoft.com/en-us/download/details.aspx?id=4422)

### Mac OS
"On Mac OS X, NUKE is built on Snow Leopard, using GCC 4.0. We recommend third-party developers to do the same."
[(NDK Dev Guide)](https://www.thefoundry.co.uk/products/nuke/developers/105/ndkdevguide/appendixa/osx.html)

Ancient compiler...

### Plugin Install Notes
On installation, the following code will be appended to $HOME/.nuke/init.py, where {CMAKE_INSTALL_PREFIX} will be the absolute path of the CMAKE_INSTALL_PREFIX variable:
```
nuke.pluginAddPath('{CMAKE_INSTALL_PREFIX}')
```
{CMAKE_INSTALL_PREFIX} defaults to: "{HOME}/.nuke/" if not provided.

Advanced:

Nuke will search for init.py in the default plugin paths.  To add an additional search path, modify the NUKE_PATH enviroment variable:
```
export NUKE_PATH=$NUKE_PATH:{location-of-init.py}
```

Additional info on [Nuke start-up scripts](https://www.thefoundry.co.uk/products/nuke/developers/105/pythondevguide/startup.html), init.py and menu.py<br>
Additional info on [installing plugins](https://www.thefoundry.co.uk/products/nuke/developers/105/pythondevguide/installing_plugins.html)

## Plugin Descriptions
AFXGlow — A beautiful glow with realistic falloff.  The extremely intuitive controls yield predictable results. [AFXGlow Youtube example](https://www.youtube.com/watch?v=nkY4S2smK_U)

AFXSoftClip — Non-linearly reduce exposure.  The algorithm was originally written for use in Eyeon Fusion.  It’s much more intuitive than the Native Nuke implementation.

AFXToneMap — Exposure control, soft clipping, and dark contrast.

AFXMedian — Extremely fast median filter with sharpness parameter to reduce unwanted morphological changes and floating point size control.  Faster than Nuke’s native median filter by an order of magnitude. [AFXMedian Youtube example](https://www.youtube.com/watch?v=SspTyatPAPg)

AFXChromaKey — Generates a 2D polygon with CIELab (A, B) Cartesian coordinates to represent the chroma screen.  Alpha is 0 for pixels within polygon, otherwise, alpha is function of distance to polygon.  Matte is invariant to lighting and grading changes due to per frame chroma analysis.

AFXDespill — Uses Rodrigues rotation to shift screen hue to either 1/3 or 2/3.  Spill suppression is is calculated using simple RGB based de-sill algorithms. Outputs a normalized spill matte.

AFXAntiAlias — Morphological anti-aliasing to smooth ‘jaggies’ that are a very common problem with keying. Extremely useful for monitor comps. [AFXAntiAlias Youtube example](https://www.youtube.com/watch?v=SspTyatPAPg)

## Redistributable Libraries
* libboost_system
* libboost_thread
* libcudart
* libippi
* libippcore
* libHalf
* libhoard

## TODO
* Write documentation
* Test on Windows and Mac OS (Built with gcc4 :( )
* Create youtube videos for suggested usage and tips
* Finish writing cufft implementation for afx_glow
* Update AFXChromaKey to use non-convex hull
* Make icons for plugins

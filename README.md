# afx-nuke-plugins

## Requirements
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [Intel IPP](https://registrationcenter.intel.com/en/forms/?productid=2558) (Free Community Licensing)
* [BOOST](https://sourceforge.net/projects/boost/files/boost/)
* Nuke NDK (Installed with Nuke)

#### Optional
* [Hoard](https://github.com/emeryberger/Hoard/releases)
* [IlmBase](http://www.openexr.com/downloads.html)

## Build and Install
```
git clone https://github.com/AuthorityFX/afx-nuke-plugins.git
cd afx-nuke-plugins
mkdir build
cd build
cmake ..
make install
```

On installation, the following code will be appended to $HOME/.nuke/init.py, where {CMAKE_INSTALL_PREFIX} will be the absolute path of the CMAKE_INSTALL_PREFIX variable:
```
nuke.pluginAddPath('{CMAKE_INSTALL_PREFIX}')
```
{CMAKE_INSTALL_PREFIX} defaults to: "{HOME}/.nuke/"

For custom install location, use:
```
cmake -DCMAKE_INSTALL_PREFIX="$HOME/.nuke/afx-nuke-plugins" ..
```

### Notes
**Linux** — "Plug-ins compiled with GCC versions 4.1 through 4.8.2 without C++11 support should be compatible...The use of C++11 via GCC, clang or the Intel compiler is untested and unsupported, especially in terms of passing standard library objects targeted at C++11 through the plug-in interface."
[(NDK Dev Guide)](https://www.thefoundry.co.uk/products/nuke/developers/105/ndkdevguide/appendixa/linux.html)<br>

**Windows** — "NUKE on Windows is compatible with plug-ins built with: *Visual Studio 2010 - (Manifest Version: 10.0.30319)*"
[(NDK Dev Guide)](https://www.thefoundry.co.uk/products/nuke/developers/105/ndkdevguide/appendixa/windows.html)<br>

**Mac OS X** — "On Mac OS X, NUKE is built on Snow Leopard, using GCC 4.0. We recommend third-party developers to do the same."
[(NDK Dev Guide)](https://www.thefoundry.co.uk/products/nuke/developers/105/ndkdevguide/appendixa/osx.html)<br>

### Advanced

Nuke will search for init.py in the default plugin paths.  To add an additional search path, modify the NUKE_PATH enviroment variable:
```
export NUKE_PATH=$NUKE_PATH:{location-of-init.py}
```

Additional info on [Nuke start-up scripts](https://www.thefoundry.co.uk/products/nuke/developers/105/pythondevguide/startup.html), init.py and menu.py<br>
Additional info on [installing plugins](https://www.thefoundry.co.uk/products/nuke/developers/105/pythondevguide/installing_plugins.html)

## Plugin Descriptions

AFXGlow — A beautiful glow with realistic falloff.  The extremely intuitive controls yield predictable results. [Youtube example](https://www.youtube.com/watch?v=nkY4S2smK_U)

AFXSoftClip — Non-linearly reduce exposure.  The algorithm was originally written for use in Eyeon Fusion.  It’s much more intuitive than the Native Nuke implementation.

AFXToneMap — Exposure control, soft clipping, and dark contrast.

AFXMedian — Extremely fast median filter with sharpness parameter to reduce unwanted morphological changes and floating point size control.  Faster than Nuke’s native median filter by an order of magnitude. [Youtube example](https://www.youtube.com/watch?v=SspTyatPAPg)

AFXChromaKey — Generates a 2D polygon with CIELab (A, B) Cartesian coordinates to represent the chroma screen.  Alpha is 0 for pixels within polygon, otherwise, alpha is function of distance to polygon.  Matte is invariant to lighting and grading changes due to per frame chroma analysis.

AFXDespill — Uses Rodrigues rotation to shift screen hue to either 1/3 or 2/3.  Spill suppression is is calculated using simple RGB based de-sill algorithms. Outputs a normalized spill matte.

AFXAntiAlias — Morphological anti-aliasing to smooth ‘jaggies’ that are a very common problem with keying. Extremely useful for monitor comps. [Youtube example](https://www.youtube.com/watch?v=SspTyatPAPg)

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
* Test on Windows and Mac OS
* Create youtube videos for suggested usage and tips
* Finish writing cufft implementation for afx_glow
* Update AFXChromaKey to use non-convex hull
* Make icons for plugins

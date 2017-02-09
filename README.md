afx-nuke-plugins
================

Requirements
------------
* Cuda
* Intel IPP (Community Licensing is free https://registrationcenter.intel.com/en/forms/?productid=2558)
* BOOST
* Nuke NDK

Build and Install
-----------------

On Linux, Cuda and Boost can be installed from your distros package management system. I perfer to download the latest version of CUDA from https://developer.nvidia.com/cuda-downloads and BOOST from https://sourceforge.net/projects/boost/files/boost/; though not required.  Intel IPP is quick to install with no addional setup required.

1. git clone https://github.com/AuthorityFX/afx-nuke-plugins.git
2. cd afx-nuke-plugins
3. mkdir build
4. cd build
5. cmake -DCMAKE_INSTALL_PREFIX="wherever-you-want/afx-nuke-plugins" ..
6. make -j{N} (where N is the num of threads)
7. make install

To tell Nuke where to find the plugins, you need to add some python code to an init.py script, or put the plugins into your Nuke home directory. https://www.thefoundry.co.uk/products/nuke/developers/63/pythondevguide/installing_plugins.html

I have a custom init.py on my network drive. In my .profile script I added "export NUKE_PATH="path to my nuke folder with the custon init.py".  In the init.py I aded:

if nuke.env['LINUX']:
        nuke.pluginAddPath(wherever-you-installed/afx-nuke-plugins)

Plugin Descriptions
===================

AFXGlow — A beautiful glow with realistic falloff.  The extremely intuitive controls yield predictable results.

AFXSoftClip — Non-linearly reduce exposure.  The algorithm was originally written for use in Eyeon Fusion.  It’s much more intuitive than the Native Nuke implementation.

AFXToneMap — Exposure control, soft clipping, and dark contrast.

AFXMedian — Extremely fast median filter with sharpness parameter to reduce unwanted morphological changes and floating point size control.  Faster than Nuke’s native median filter by an order of magnitude.

AFXChromaKey — Generates a 2D polygon with CIELab (A, B) Cartesian coordinates to represent the chroma screen.  Alpha is 0 for pixels within polygon, otherwise, alpha is function of distance to polygon.  Matte is invariant to lighting and grading changes due to per frame chroma analysis.

AFXDespill — Uses Rodrigues rotation to shift screen hue to either 1/3 or 2/3.  Spill suppression is is calculated using simple RGB based de-sill algorithms. Outputs a normalized spill matte.

AFXAntiAlias — Morphological anti-aliasing to smooth ‘jaggies’ that are a very common problem with keying. Extremely useful for monitor comps.


Redistributable Libraries
-------------------------

* libboost_system.so
* libboost_thread.so
* libcudart.so
* libippcore.so
* libippie9.so
* libippik0.so
* libippil9.so
* libippim7.so
* libippimx.so
* libippin0.so
* libippin8.so
* libippi.so
* libippiy8.so

TODO
--------------------------------------------------

* Write documentation
* Create youtube videos for suggested usage and tips
* Add expand border bbox option to afx_glow
* Finish writing cufft implementation for afx_glow
* Update AFXChromaKey to use non-convex hull and kdtree search
* Polish cmake
* Improve mlaa algorithm
* Make icons for plugins
* Add MacOS support to CMakeLists.txt

If you like these plugins, help me make them better. Help me write more!
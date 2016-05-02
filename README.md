afx_nuke_plugins
================

Requirements
------------
* Cuda
* Intel IPP (Community Licensing is free https://registrationcenter.intel.com/en/forms/?productid=2558)
* BOOST
* Nuke NDK

Install
-------
* git clone https://github.com/AuthorityFX/afx_nuke_plugins.git
* cd afx_nuke_plugins
* mkdir build
* cd build
* cmake -DCMAKE_INSTALL_PREFIX="wherever-you-want/afx_nuke_plugins" ..
* make -j{N} (where N is the num of threads)
* make install

Add to init.py in your nuke directory

if nuke.env['LINUX']:
        nuke.pluginAddPath(wherever-you-installed/afx_nuke_plugins)

Plugin Descriptions
===================

AFXSoftClip — Non-linearly reduce exposure.  The algorithm was originally written for use in Eyeon Fusion.  It’s much more intuitive than the Native Nuke implementation.

AFXToneMap — Exposure control, soft clipping, and dark contrast.

AFXMedian — Extremely fast median filter with sharpness parameter to reduce unwanted morphological changes and floating point size control.  Faster than Nuke’s native median filter by an order of magnitude.

AFXChromaKey — Performs statistical analysis of chroma screen in LAB color space.  Easily generate solid core mattes and detailed edge mattes.  Invariant to lighting and grading changes due to per frame chroma analysis.

AFXDespill — Simply the best de-spill available.  Output a spill matte to use as a mask for replacing de-spilled areas with a blurred version of the background or a reflection pass.

AFXAntiAlias — Morphological anti-aliasing to smooth ‘jaggies’ that are a very common problem with keying. Extremely useful for monitor comps.

AFXGlow — A beautiful glow.

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

* Add expand border bbox option to afx_glow
* Finish writing cufft implementation for afx_glow
* Write afx_keyer 2.0 — notes are in afx_chrome_key.cpp
* Polish cmake
* Write montecarlo noise removal tools
* Improve mlaa algorithm
* Make icons for plugins

If you like these plugins, help me make them better. Help me write more!
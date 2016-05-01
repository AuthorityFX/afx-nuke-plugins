afx_nuke_plugins
================

Requirements
------------
* Cuda 7.5
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
* make -j{N} where N is the num of threads
* make install

Add to init.py in your nuke directory

if nuke.env['LINUX']:
        nuke.pluginAddPath(wherever-you-installed/afx_nuke_plugins)


Redistributable Libraries
-------------------------

* libboost_system.so.1.60.0
* libboost_thread.so.1.60.0
* libcudart.so.7.5
* libippcore.so.9.0
* libippie9.so.9.0
* libippik0.so.9.0
* libippil9.so.9.0
* libippim7.so.9.0
* libippimx.so.9.0
* libippin0.so.9.0
* libippin8.so.9.0
* libippi.so.9.0
* libippiy8.so.9.0

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
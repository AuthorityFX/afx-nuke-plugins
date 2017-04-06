# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (C) 2012-2016, Ryan P. Wilson
#
#      Authority FX, Inc.
#      www.authorityfx.com


afx_plugin_path = os.path.dirname(os.path.realpath(__file__))
nuke.pluginAddPath(os.path.join(afx_plugin_path, 'icons'))
nuke.menu('Nodes').addMenu('Authority FX', icon='afx.png')

lib_extension = '.so'
if os.name == 'nt':
    lib_extension = '.dll'
    # Windows has no RPATH equivalent - need to append lib dir to PATH variable
    os.environ['PATH'] += os.pathsep + os.path.join(afx_plugin_path, 'lib')

if os.path.isfile(os.path.join(afx_plugin_path, 'afx_soft_clip' + lib_extension)):
    nuke.menu('Nodes').addCommand('Authority FX/AFXSoftClip', lambda: nuke.createNode('AFXSoftClip'), icon='afx.png')
    nuke.load('afx_soft_clip' + lib_extension)
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_tone_map' + lib_extension)):
    nuke.menu('Nodes').addCommand('Authority FX/AFXToneMap', lambda: nuke.createNode('AFXToneMap'), icon='afx.png')
    nuke.load('afx_tone_map' + lib_extension)
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_median' + lib_extension)):
    nuke.knobDefault('AFXMedian.channels', 'alpha')
    nuke.menu('Nodes').addCommand('Authority FX/AFXMedian', lambda: nuke.createNode('AFXMedian'), icon='afx.png')
    nuke.load('afx_median' + lib_extension)
# Noise map is only partially implemented
#if os.path.isfile(os.path.join(afx_plugin_path, 'afx_noise_map' + lib_extension)):
    #nuke.menu('Nodes').addCommand('Authority FX/AFXNoiseMap', lambda: nuke.createNode('AFXNoiseMap'), icon='afx.png')
    #nuke.load('afx_noise_map' + lib_extension)
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_chroma_key' + lib_extension)):
    nuke.menu('Nodes').addCommand('Authority FX/AFXChromaKey', lambda: nuke.createNode('AFXChromaKey'), icon='afx.png')
    nuke.load('afx_chroma_key' + lib_extension)
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_despill' + lib_extension)):
    nuke.menu('Nodes').addCommand('Authority FX/AFXDeSpill', lambda: nuke.createNode('AFXDeSpill'), icon='afx.png')
    nuke.load('afx_despill' + lib_extension)
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_anti_alias' + lib_extension)):
    nuke.menu('Nodes').addCommand('Authority FX/AFXAntiAlias', lambda: nuke.createNode('AFXAntiAlias'), icon='afx.png')
    nuke.load('afx_anti_alias' + lib_extension)
    nuke.knobDefault('AFXAntiAlias.channels', 'alpha')
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_glow' + lib_extension)):
    nuke.menu('Nodes').addCommand('Authority FX/AFXGlow', lambda: nuke.createNode('AFXGlow'), icon='afx.png')
    nuke.load('afx_glow' + lib_extension)

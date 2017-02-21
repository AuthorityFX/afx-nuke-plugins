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

if os.path.isfile(os.path.join(afx_plugin_path, 'afx_soft_clip.so')):
    nuke.menu('Nodes').addCommand('Authority FX/AFXSoftClip', lambda: nuke.createNode('AFXSoftClip'), icon='afx.png')
    nuke.load('afx_soft_clip.so')
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_tone_map.so')):
    nuke.menu('Nodes').addCommand('Authority FX/AFXToneMap', lambda: nuke.createNode('AFXToneMap'), icon='afx.png')
    nuke.load('afx_tone_map.so')
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_median.so')):
    nuke.knobDefault('AFXMedian.channels', 'alpha')
    nuke.menu('Nodes').addCommand('Authority FX/AFXMedian', lambda: nuke.createNode('AFXMedian'), icon='afx.png')
    nuke.load('afx_median.so')
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_chroma_key.so')):
    nuke.menu('Nodes').addCommand('Authority FX/AFXChromaKey', lambda: nuke.createNode('AFXChromaKey'), icon='afx.png')
    nuke.load('afx_chroma_key')
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_despill.so')):
    nuke.menu('Nodes').addCommand('Authority FX/AFXDeSpill', lambda: nuke.createNode('AFXDeSpill'), icon='afx.png')
    nuke.load('afx_despill.so')
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_anti_alias.so')):
    nuke.menu('Nodes').addCommand('Authority FX/AFXAntiAlias', lambda: nuke.createNode('AFXAntiAlias'), icon='afx.png')
    nuke.load('afx_anti_alias.so')
    nuke.knobDefault('AFXAntiAlias.channels', 'alpha')
if os.path.isfile(os.path.join(afx_plugin_path, 'afx_glow.so')):
    nuke.menu('Nodes').addCommand('Authority FX/AFXGlow', lambda: nuke.createNode('AFXGlow'), icon='afx.png')
    nuke.load('afx_glow.so')

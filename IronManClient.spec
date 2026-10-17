# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

# Asset directories to bundle inside the executable package
added_datas = [
    ('local_client/cool_graphics', 'cool_graphics'),
    ('local_client/audio_files', 'audio_files'),
]

# Exclude massive backend training frameworks to keep the client exe sleek & lightweight
excluded_modules = [
    'torch',
    'torchvision',
    'torch_directml',
    'scipy',
    'sklearn',
    'pandas',
    'matplotlib',
    'fastapi',
    'uvicorn',
    'pydantic',
    'sympy',
    'mpmath',
    'IPython',
]

a = Analysis(
    ['local_client/capture.py'],
    pathex=['local_client'],
    binaries=[],
    datas=added_datas,
    hiddenimports=[
        'PIL._tkinter_finder',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Builds high-performance distribution folder: dist/IronManClient/IronManClient.exe
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='IronManClient',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='local_client/cool_graphics/IronManIcon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='IronManClient',
)

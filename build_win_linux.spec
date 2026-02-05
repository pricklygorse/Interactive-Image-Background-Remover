# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all 

datas_pymatting, binaries_pymatting, hiddenimports_pymatting = collect_all('pymatting')

a = Analysis(
    ['backgroundremoval.py'],
    pathex=[],
    binaries=binaries_pymatting,
    datas=datas_pymatting, 
    hiddenimports=hiddenimports_pymatting, 
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch','PyQt5'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
splash = Splash(
    'src/assets/splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10, 30),
    text_size=12,
    text_color='black',
    minify_script=True,
    always_on_top=False,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    [],
    exclude_binaries=True,
    name='Interactive-BG-Remover',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    splash.binaries,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Interactive-BG-Remover',
)

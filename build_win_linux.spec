# -*- mode: python ; coding: utf-8 -*-
#from PyInstaller.utils.hooks import collect_all 

# datas_pymatting, binaries_pymatting, hiddenimports_pymatting = collect_all('pymatting')
# datas_ort, binaries_ort, hidden_ort = collect_all('onnxruntime')

a = Analysis(
    ['backgroundremoval.py'],
    pathex=[],
    # binaries=binaries_pymatting + binaries_ort,
    # datas=datas_pymatting + datas_ort, 
    # hiddenimports=hiddenimports_pymatting + hidden_ort, 
    binaries=[],
    datas=[], 
    hiddenimports=[], 
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch','PyQt5'],
    noarchive=False,
    optimize=0,
)

# onnxruntime seems to try using pyqt6's bundled visual studio runtime in newest github runner feb 2026
# so we remove it and let pyinstaller handle using the system dll
new_binaries = []
for (dest, source, kind) in a.binaries:
    # Check if the binary is a VC runtime and if it comes from PyQt6
    if "MSVCP140" in dest.upper() or "VCRUNTIME140" in dest.upper():
        if "PyQt6" in source:
            print(f"Removing conflicting DLL from bundle: {source}")
            continue
    new_binaries.append((dest, source, kind))

a.binaries = new_binaries


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
